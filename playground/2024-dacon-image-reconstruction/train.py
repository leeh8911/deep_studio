"""
"""

from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description="2024-dacon-image-reconstruction")
    parser.add_argument("--config", help="train config file")

def main():
    