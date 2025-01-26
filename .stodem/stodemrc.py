#!/usr/bin/env python3


def parameters_and_defaults():
    param_dict = {
            "infile" : "stodem.in.xml", # String
            "outfile" : "stodem" # String
            }
    return param_dict


if __name__ == '__main__':
    print(parameters_and_defaults())
