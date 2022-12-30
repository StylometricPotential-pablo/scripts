

import os

def analyze_docuemnt(path):
    if (not os.exists(path)):
        return None

    return (avg_wordlength(path), avg_sentencelength(path))


def avg_wordlength(path):
    return 


def avg_sentencelength(path):
    return
