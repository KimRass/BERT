from time import time
from datetime import timedelta


def get_elapsed_time(start_time):
    return timedelta(seconds=round(time() - start_time))

def print_number_of_parameters(model):
    print(f"""{sum([p.numel() for p in model.parameters()]):,}""")
