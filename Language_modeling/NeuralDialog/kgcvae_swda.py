## main function

import os
import time

import numpy as np
import tensorflow as tf
from beeprint import pp

from config_utils import KgCVAEConfig as Config
from data_apis.corpus import SWDADialogCorpus
from data_apis.data_utils import SWDADataLoader
from models.cvae import KgRnnCVAE
from tqdm import tqdm
from PID import PIDControl
# import numpy as np


# constants
tf.app.flags.DEFINE_string("word2vec_path", './glove/glove.twitter.27B.200d.txt', "The path to word2vec. Can be None.")
tf.app.flags.DEFINE_string("data_dir", "./data/full_swda_clean_42da_sentiment_dialog_corpus.p", "Raw data directory.")
tf.app.flags.DEFINE_string("work_dir", "working", "checkpoint results directory.")
tf.app.flags.DEFINE_bool("equal_batch", True, "Make each batch has similar length.")
tf.app.flags.DEFINE_bool("resume", False, "Resume from previous")
tf.app.flags.DEFINE_bool("forward_only", False, "Only do decoding")
tf.app.flags.DEFINE_string("model_name", "cost_anneal", "cost annealing model.")
tf.app.flags.DEFINE_string("gpu", "1", "gpu id.")
tf.app.flags.DEFINE_string("test_res", "test_1", "name of test file.")
tf.app.flags.DEFINE_string("mode", "train", "train/valid/test.")
tf.app.flags.DEFINE_bool("save_model", True, "Create checkpoints")
tf.app.flags.DEFINE_float('exp_KL', 30, 'expected KL divergence')
tf.app.flags.DEFINE_float('Kp', 0.01, 'P term for PID')
tf.app.flags.DEFINE_float('Ki', -0.0001, 'I term for PID')
tf.app.flags.DEFINE_integer('cycle', 28950, 'cycles for cyclical annealing')
tf.app.flags.DEFINE_integer('anneal_steps', 20000, 'cost annealing')
tf.app.flags.DEFINE_integer('seed', 1, 'seed for random')
tf.app.flags.DEFINE_string("test_path", "run_cost_anneal_20000", "the dir to load checkpoint for forward only")

FLAGS = tf.app.flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
print('forward_only: ', FLAGS.forward_only)


def main():
    ## random seeds
    seed = FLAGS.seed
    # tf.random.set_seed(seed)
    np.random.seed(seed)

    ## config for training
    config = Config()
    pid = PIDControl(FLAGS.exp_KL)
    
    # config for validation
    valid_config = Config()
    valid_config.keep_prob = 1.0
    valid_config.dec_keep_prob = 1.0
    valid_config.batch_size = 60

    # configuration for testing
    test_config = Config()
    test_config.keep_prob = 1.0
    test_config.dec_keep_prob = 1.0
    test_config.batch_size = 1

    pp(config)

    # get data set
    api = SWDADialogCorpus(FLAGS.data_dir, word2vec=FLAGS.word2vec_path, word2vec_dim=config.embed_size)
    dial_corpus = api.get_dialog_corpus()
    meta_corpus = api.get_meta_corpus()

    train_meta, valid_meta, test_meta = meta_corpus.get("train"), meta_corpus.get("valid"), meta_corpus.get("test")
    train_dial, valid_dial, test_dial = dial_corpus.get("train"), dial_corpus.get("valid"), dial_corpus.get("test")
    
    # convert to numeric input outputs that fits into TF models
    train_feed = SWDADataLoader("Train", train_dial, train_meta, config)
    valid_feed = SWDADataLoader("Valid", valid_dial, valid_meta, config)
    test_feed = SWDADataLoader("Test", test_dial, test_meta, config)

    if FLAGS.forward_only or FLAGS.resume:
        # log_dir = os.path.join(FLAGS.work_dir, FLAGS.test_path)
        log_dir = os.path.join(FLAGS.work_dir, FLAGS.model_name)
    else:
        log_dir = os.path.join(FLAGS.work_dir, FLAGS.model_name)

    
    ## begin training
    with tf.Session() as sess:
        initializer = tf.random_uniform_initializer(-1.0 * config.init_w, config.init_w)
        scope = "model"
        with tf.variable_scope(scope, reuse=None, initializer=initializer):
            model = KgRnnCVAE(sess, config, api, log_dir=None if FLAGS.forward_only else log_dir, forward=False, pid_control=pid, scope=scope)
        with tf.variable_scope(scope, reuse=True, initializer=initializer):
            valid_model = KgRnnCVAE(sess, valid_config, api, log_dir=None, forward=False, pid_control=pid, scope=scope)
        with tf.variable_scope(scope, reuse=True, initializer=initializer):
            test_model = KgRnnCVAE(sess, test_config, api, log_dir=None, forward=True, pid_control=pid, scope=scope)

        print("Created computation graphs")
        if api.word2vec is not None and not FLAGS.forward_only:
            print("Loaded word2vec")
            sess.run(model.embedding.assign(np.array(api.word2vec)))

        # write config to a file for logging
        if not FLAGS.forward_only:
            with open(os.path.join(log_dir, "configure.log"), "wb") as f:
                f.write(pp(config, output=False))
        
        # create a folder by force
        ckp_dir = os.path.join(log_dir, "checkpoints")
        print("*******checkpoint path: ", ckp_dir)
        if not os.path.exists(ckp_dir):
            os.mkdir(ckp_dir)

        ckpt = tf.train.get_checkpoint_state(ckp_dir)
        print("Created models with fresh parameters.")
        sess.run(tf.global_variables_initializer())

        if ckpt:
            print("Reading dm models parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        ### save log when running
        if not FLAGS.forward_only:
            logfileName = "train.log"
        else:
            logfileName = "test.log"
        fw_log = open(os.path.join(log_dir, logfileName), "w")
        print("log directory >>> : ", os.path.join(log_dir, "run.log"))
        if not FLAGS.forward_only:
            print('--start training now---')
            dm_checkpoint_path = os.path.join(ckp_dir, model.__class__.__name__+ ".ckpt")
            global_t = 1
            patience = 20  # wait for at least 10 epoch before stop
            dev_loss_threshold = np.inf
            best_dev_loss = np.inf
            pbar = tqdm(total = config.max_epoch)
            ## epoch start training
            for epoch in range(config.max_epoch):
                pbar.update(1)
                print(">> Epoch %d with lr %f" % (epoch, model.learning_rate.eval()))

                ## begin training
                FLAGS.mode = 'train'
                if train_feed.num_batch is None or train_feed.ptr >= train_feed.num_batch:
                    train_feed.epoch_init(config.batch_size, config.backward_size,
                                          config.step_size, shuffle=True)
                global_t, train_loss = model.train(global_t, sess, train_feed, update_limit=config.update_limit)
                
                FLAGS.mode = 'valid'
                valid_feed.epoch_init(valid_config.batch_size, valid_config.backward_size,
                                  valid_config.step_size, shuffle=False, intra_shuffle=False)
                test_feed.epoch_init(valid_config.batch_size, valid_config.backward_size,
                                  valid_config.step_size, shuffle=False, intra_shuffle=False)
                elbo, nll, ppl, au_count, kl_loss = valid_model.valid("ELBO_TEST", sess, valid_feed, test_feed)
                print('middle test nll: {} ppl: {} ActiveUnit: {} kl_loss:{}\n'.format(nll, ppl,au_count,kl_loss))
                fw_log.write('epoch:{} testing nll:{} ppl:{} ActiveUnit:{} kl_loss:{} elbo:{}\n'.\
                            format(epoch, nll, ppl, au_count, kl_loss, elbo))
                fw_log.flush()
                
                '''
                ## begin validation
                FLAGS.mode = 'valid'
                valid_feed.epoch_init(valid_config.batch_size, valid_config.backward_size,
                                      valid_config.step_size, shuffle=False, intra_shuffle=False)
                valid_loss = valid_model.valid("ELBO_VALID", sess, valid_feed)

                ## test model
                FLAGS.mode = 'test'
                test_feed.epoch_init(test_config.batch_size, test_config.backward_size,
                                     test_config.step_size, shuffle=True, intra_shuffle=False)
                test_model.test(sess, test_feed, num_batch=5)

                done_epoch = epoch + 1
                # only save a models if the dev loss is smaller
                # Decrease learning rate if no improvement was seen over last 3 times.
                if config.op == "sgd" and done_epoch > config.lr_hold:
                    sess.run(model.learning_rate_decay_op)

                if valid_loss < best_dev_loss:
                    if valid_loss <= dev_loss_threshold * config.improve_threshold:
                        patience = max(patience, done_epoch * config.patient_increase)
                        dev_loss_threshold = valid_loss

                    # still save the best train model
                    if FLAGS.save_model:
                        print("Save model!!")
                        model.saver.save(sess, dm_checkpoint_path, global_step=epoch)
                    best_dev_loss = valid_loss

                if config.early_stop and patience <= done_epoch:
                    print("!!Early stop due to run out of patience!!")
                    break
                    ## print("Best validation loss %f" % best_dev_loss)
                 '''
            print("Done training and save checkpoint")

            if FLAGS.save_model:
                print("Save model!!")
                model.saver.save(sess, dm_checkpoint_path, global_step=epoch)
            # begin validation
            print('--------after training to testing now-----')
            FLAGS.mode = 'test'
            # valid_feed.epoch_init(valid_config.batch_size, valid_config.backward_size,
                                #   valid_config.step_size, shuffle=False, intra_shuffle=False)
            # valid_model.valid("ELBO_VALID", sess, valid_feed)
            valid_feed.epoch_init(valid_config.batch_size, valid_config.backward_size,
                                  valid_config.step_size, shuffle=False, intra_shuffle=False)
            
            test_feed.epoch_init(valid_config.batch_size, valid_config.backward_size,
                                  valid_config.step_size, shuffle=False, intra_shuffle=False)
            elbo, nll, ppl, au_count,kl_loss = valid_model.valid("ELBO_TEST", sess, valid_feed, test_feed)

            print('final test nll: {} ppl: {} ActiveUnit: {} kl_loss:{}\n'.format(nll, ppl,au_count,kl_loss))
            fw_log.write('Final testing nll:{} ppl:{} ActiveUnit:{} kl_loss:{} elbo:{}\n'.\
                            format(nll, ppl, au_count, kl_loss, elbo))
            
            dest_f = open(os.path.join(log_dir, FLAGS.test_res), "wb")
            test_feed.epoch_init(test_config.batch_size, test_config.backward_size,
                                 test_config.step_size, shuffle=False, intra_shuffle=False)
            test_model.test(sess, test_feed, num_batch=None, repeat=10, dest=dest_f)
            dest_f.close()
            print("****testing done****")
        else:
            # begin validation
            # begin validation
            print('*'*89)
            print('--------testing now-----')
            print('*'*89)
            FLAGS.mode = 'test'
            valid_feed.epoch_init(valid_config.batch_size, valid_config.backward_size,
                                  valid_config.step_size, shuffle=False, intra_shuffle=False)
            # valid_model.valid("ELBO_VALID", sess, valid_feed)

            test_feed.epoch_init(valid_config.batch_size, valid_config.backward_size,
                                  valid_config.step_size, shuffle=False, intra_shuffle=False)
            elbo, nll, ppl, au_count, kl_loss = valid_model.valid("ELBO_TEST", sess, valid_feed, test_feed)

            print('final test nll: {} ppl: {} ActiveUnit: {} kl_loss:{}\n'.format(nll, ppl,au_count,kl_loss))
            fw_log.write('Final testing nll:{} ppl:{} ActiveUnit:{} kl_loss:{} elbo:{}\n'.\
                            format(nll, ppl, au_count, kl_loss, elbo))
            # dest_f = open(os.path.join(log_dir, FLAGS.test_res), "wb")
            # test_feed.epoch_init(test_config.batch_size, test_config.backward_size,
            #                      test_config.step_size, shuffle=False, intra_shuffle=False)
            # test_model.test(sess, test_feed, num_batch=None, repeat=10, dest=dest_f)
            # dest_f.close()
            print("****testing done****")
        fw_log.close()
        

if __name__ == "__main__":
    if FLAGS.forward_only:
        if FLAGS.test_path is None:
            print("Set test_path before forward only")
            exit(1)
    main()
    












