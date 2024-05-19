import torch

deepseek_vl_1_3b = torch.tensor([1307+225, 64.6, 34.8, 51.1, 75.0, 62.8, 68.2, 64.9, 63.4, 68.3])
mgm_2b = torch.tensor([1341+312, 59.8, 31.1, 65.9, 75.0, 63.7, 67.3, 65.6, 64.4, 68.4])
llava_1_5_7b = torch.tensor([1511+348, 64.3, 30.5, 69.0, 75.2, 63.7, 67.1, 64.8, 63.4, 68.2])
hpt_air_7b = torch.tensor([1010+258, 69.8, 31.3, 59.2, 74.3, 64.0, 67.5, 65.5, 64.0, 68.8])
hpt_air_1_5_8b = torch.tensor([1476+308, 75.2, 36.3, 62.1, 76.3, 64.5, 68.5, 65.4, 64.1, 68.5])
mgm_7b = torch.tensor([1523+316, 69.3, 40.8, 75.8, 75.7, 64.8, 68.3, 66.3, 65.3, 68.6])
deepseek_vl_7b = torch.tensor([1468+298, 73.2, 41.5, 77.8, 76.1, 66.4, 70.1, 65.7, 64.5, 68.5])
llava_1_6_7b = torch.tensor([1519/322, 68.1, 44.1, 72.3, 75.8, 65.8, 70.1, 66.3, 65.1, 69.0])
llava_1_6_m_7b = torch.tensor([1501+324, 69.5, 47.8, 71.7, 75.7, 66.5, 70.1, 66.5, 65.4, 69.1])
mgm_hd_7b = torch.tensor([1546+319, 65.8, 41.3, 74.0, 76.1, 65.2, 68.5, 66.7, 65.6, 69.1])

all_scores = torch.stack([deepseek_vl_1_3b, mgm_2b,
                          llava_1_5_7b, hpt_air_7b, hpt_air_1_5_8b, mgm_7b, deepseek_vl_7b,
                          llava_1_6_7b, llava_1_6_m_7b, mgm_hd_7b])

all_ranks = torch.sort(-all_scores, dim=0).indices


qa_ranks = all_ranks[:, :4]
seg_ranks = all_ranks[:, 4:]


qa_ave_ranks = qa_ranks.float().mean(dim=-1)
seg_ave_ranks = seg_ranks.float().mean(dim=-1)


import matplotlib.pyplot as plt
plt.scatter((10 - qa_ave_ranks).tolist(), (10 - seg_ave_ranks).tolist())
plt.savefig('ave_ranks.jpg')

