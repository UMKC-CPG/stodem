import argparse as ap
import os
import sys
from datetime import datetime

from lxml import etree


class ScriptSettings():
    """The instance variables of this object are the user settings that
       control the program. The variable values are pulled from a list
       that is created within a resource control file and that are then
       reconciled with command line parameters."""


    def __init__(self):
        """Define default values for the graph parameters by pulling them
        from the resource control file in the default location:
        $STODEM_RC/stodemrc.py or from the current working directory if a local
        copy of stodemrc.py is present."""

        # Read default variables from the resource control file.
        sys.path.insert(1, os.getenv('STODEM_RC'))
        from stodemrc import parameters_and_defaults
        default_rc = parameters_and_defaults()

        # Assign values to the settings from the rc defaults file.
        self.assign_rc_defaults(default_rc)

        # Parse the command line.
        args = self.parse_command_line()

        # Reconcile the command line arguments with the rc file.
        self.reconcile(args)

        # At this point, the command line parameters are set and accepted.
        #   When this initialization subroutine returns the script will
        #   start running. So, we use this as a good spot to record the
        #   command line parameters that were used.
        self.recordCLP()


    def assign_rc_defaults(self, default_rc):

        # Default filename variables.
        self.infile = default_rc["infile"]
        self.outfile = default_rc["outfile"]


    def parse_command_line(self):

        # Create the parser tool.
        prog_name = "stodem"

        description_text = """
Version 0.1
The purpose of this program is to simulate the effect of stochastic voting
on the ability of a democracy to navigate a high-dimensional policy space
and find and exploit the global minimum. The global minimum is the point
with the strongest alignment between the internalized policy preferences of
a population (and its politicians) and the actual (unknown) policies that
lead to positive outcomes for the population.
"""

        epilog_text = """
Please contact Paul Rulis (rulisp@umkc.edu) regarding questions.
Defaults are given in $STODEM_RC/stodemrc.py.
"""

        parser = ap.ArgumentParser(prog = prog_name,
                formatter_class=ap.RawDescriptionHelpFormatter,
                description = description_text, epilog = epilog_text)

        # Add arguments to the parser.
        self.add_parser_arguments(parser)

        # Parse the arguments and return the results.
        return parser.parse_args()


    def add_parser_arguments(self, parser):

        # Define the input file.
        parser.add_argument('-i', '--infile', dest='infile', type=ascii,
                            default=self.infile, help='Input file name. ' +
                            f'Default: {self.infile}')

        # Define the output file prefix.
        parser.add_argument(
            '-o', '--outfile',
            dest='outfile', type=ascii,
            default=self.outfile,
            help='Output file name prefix for '
                 'hdf5 and xdmf. '
                 f'Default: {self.outfile}')

        # Enable the debug policy/trait-space
        #   visualization (DESIGN §12.6). Opens a
        #   live pyqtgraph window showing 2-D
        #   projected Gaussian curves with colour
        #   saturation encoding engagement.
        parser.add_argument(
            '-d', '--debug-viz',
            dest='debug_viz',
            action='store_true',
            default=False,
            help='Enable debug visualization of '
                 'policy/trait space.')

        # Minimum pause between frames (seconds).
        #   Throttles the live display so the
        #   developer can watch state evolve.
        #   Larger values slow the animation;
        #   smaller values let it run closer to
        #   simulation speed.
        parser.add_argument(
            '--viz-delay',
            dest='viz_delay', type=float,
            default=0.1,
            help='Seconds between viz frames. '
                 'Default: 0.1')


    def reconcile(self, args):
        self.infile = args.infile.strip("'")
        self.outfile = args.outfile.strip("'")
        self.debug_viz = args.debug_viz
        self.viz_delay = args.viz_delay


    def recordCLP(self):
        with open("command", "a") as cmd:
            now = datetime.now()
            formatted_dt = now.strftime("%b. %d, %Y: %H:%M:%S")
            cmd.write(f"Date: {formatted_dt}\n")
            cmd.write(f"Cmnd:")
            for argument in sys.argv:
                cmd.write(f" {argument}")
            cmd.write("\n\n")


    def read_input_file(self):

        tree = etree.parse(self.infile)
        root = tree.getroot()

        def recursive_dict(element):
            return (element.tag,
                    dict(map(recursive_dict, element)) or element.text)

        self.infile_dict = recursive_dict(root)
