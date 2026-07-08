#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os

# Mudar para o diretório correto
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Executar o script benchmark
exec(open('benchmark_simple.py').read())
