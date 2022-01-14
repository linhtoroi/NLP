from pickle import TRUE
from src.NL2SQL import Params, NL2SQLmodel
import os
import torch
import faulthandler; faulthandler.enable()
import resource, sys
resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
sys.setrecursionlimit(10**6)

# If run improvement model set improvement=True:
args = Params(improvement=True)

os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.gpu)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.model_id = 2
assert(args.model_id is not None)
args.train = True

model = NL2SQLmodel(args)
model.model.cuda()
model.train()

