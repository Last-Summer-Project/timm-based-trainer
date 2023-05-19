import timm
import pprint

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(timm.list_models(pretrained=True))