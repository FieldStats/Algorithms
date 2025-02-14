import torch
from torchvision.ops import nms

# Test data
boxes = torch.tensor([[0, 0, 100, 100], [10, 10, 110, 110]], dtype=torch.float32).cuda()
scores = torch.tensor([0.9, 0.8]).cuda()

try:
    keep = nms(boxes, scores, 0.5)
    print("NMS success:", keep)
except Exception as e:
    print("NMS failed:", e)
