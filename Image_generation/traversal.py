"""
Fun: z hidden state traversal for testing
"""

def _rand_z(num=5,decoder):
    z = torch.randn(num*num, )
    z = Variable(z, volatile=True)
    z = z.cuda()
    recon = F.sigmoid(decoder(z)).data
    torchvision.utils.save_image(recon.data, '../imgs/rand_faces.jpg', nrow=num, padding=2)
    