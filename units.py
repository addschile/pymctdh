### global constants ###
hbar = None

### dictionary for hbar ###
hbars = {'ev': 0.658229,
         'cm': 5308.8,
         'au': 1.0
        }

### energy units ###
conv = {'ev': 0.0367493,
        'cm': 4.556e-6,
        'au': 1.0,
        'fs': 41.3413745758,
        'ps': 41.3413745758*1000.}

### mass units ###
me2au = 0.00054858 # amu/m_e

### distance units ###
ang2bohr = 1.88973 # bohr/ang

def convert_to(unit):
    """
    """
    return conv[unit]

def convert_from(unit):
    """
    """
    return 1./conv[unit]

