'''
Utility functions to compute metrics
'''

def parameter_count(model):
    '''
    Returns number of parameters in a model
    '''
    return sum(p.numel() for p in model.parameters())
