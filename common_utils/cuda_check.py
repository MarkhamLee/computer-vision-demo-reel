import sys
import os
import torch

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from common_utils.logging_utils import LoggingUtilities  # noqa: E401

logger = LoggingUtilities.console_out_logger('cuda')

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logger.info(f'Running on device: {device}')
