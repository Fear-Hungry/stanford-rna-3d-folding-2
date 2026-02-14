from __future__ import annotations

import argparse

from .cli_parser_data import register_data_parsers
from .cli_parser_gating import register_gating_parsers
from .cli_parser_qa import register_qa_parsers
from .cli_parser_templates import register_template_parsers, register_template_post_qa_parsers


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rna3d_local", add_help=True)
    sp = p.add_subparsers(dest="cmd", required=True)

    register_data_parsers(sp)
    register_template_parsers(sp)
    register_qa_parsers(sp)
    register_template_post_qa_parsers(sp)
    register_gating_parsers(sp)

    return p
