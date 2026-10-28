import os,sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.strategy.micro_scalp_20_paper_runner import MicroScalp20PaperEngine
from src.strategy.micro_scalp_20 import recipe_banner_lines
print(chr(10).join(recipe_banner_lines()))
print(" import_ok\)
