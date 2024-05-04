import sys
import os
import torch

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils.logging_utils import logger  # noqa: E401
# from utils.com_utilities import CommUtilities

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logger.info(f'Running on device: {device}')
