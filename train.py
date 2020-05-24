##############################################
# The functions used to train the model.
# 
# Created 24 Sep. 2019.
# By Benxin ZHONG

import numpy as np
import sys
import time
import numpy as np
import argparse
from import_data import import_data
#from psc_funcs import SkltDrawer
#from psc_funcs import Trace
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import row
from bokeh.models import Arrow, OpenHead, NormalHead, VeeHead


def main():