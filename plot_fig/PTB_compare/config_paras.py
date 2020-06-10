# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""VAE config.
"""


path_list = ['pid_ptb_KL1.5','pid_ptb_KL3.0',\
				'cost_anneal_b32_ptb_step10000.0', 'cost_anneal_b32_ptb_step20000.0',\
				'cyclical_b32_ptb_cyc_4.0', 'cyclical_b32_ptb_cyc_8.0']
## label
label_lst =['ControlVAE-KL-1.5', 'ControlVAE-KL-3',\
			'KL-anneal-10K','KL-anneal-20K','cyclical-4','cyclical-8']

## line colors
colors = ['blue', 'red','lime','black','darkgreen','magenta', 'lime',\
				'fuchsia','blue','grey','pink','grey','coral','magenta']
