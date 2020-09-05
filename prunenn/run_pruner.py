'''
Provides functions to utilize the Pruner class and run high level training algorithms
'''

from .trainer import train, test

def retrain(pruner_model, loaders, optimizer, epoch_range=1, device=None):
    '''
    Train model in pruner
    '''
    train_loader, test_loader = loaders
    for epoch in range(epoch_range):
        pruner_model.to_train = True
        train(pruner_model, train_loader, optimizer, epoch, device)
        pruner_model.to_train = False
        acc = test(pruner_model, test_loader, device)
        pruner_model.to_train = True
    return acc

def prune_loop(pruner_model, thresholds, sacrifice, loaders, optimizer,
               retries_after_accuracy_drop=0, device=None):
    '''
    Pruning loop
    '''
    _, test_loader = loaders
    init_model_acc = test(pruner_model, test_loader)

    for threshold in thresholds:
        print("--- threshold ", threshold, " ---")
        pruner_model.set_threshold(threshold)
        retrain(pruner_model, loaders, optimizer, device=device)

        pruner_model.to_train = False
        pruner_model.prune()
        pruner_model.to_train = True
        pruning_model_acc = test(pruner_model, test_loader)

        retries = 0
        for retries in range(retries_after_accuracy_drop):
            if(pruning_model_acc <= init_model_acc - sacrifice):
                print("--- accuracy drop ", retries, " ---")
                retrain(pruner_model, loaders, optimizer, device=device)
                pruning_model_acc = test(pruner_model, test_loader)

        if pruning_model_acc <= init_model_acc - sacrifice:
            return model
        model = pruner_model.model

    return model
