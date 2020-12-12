#!/usr/bin/python

import sys
from json import loads

from tools.trainer import Trainer

if __name__ == '__main__':
    assert len(sys.argv) >= 2
    Trainer(**loads(open(f'config/{sys.argv[1]}.json').read())).run()
