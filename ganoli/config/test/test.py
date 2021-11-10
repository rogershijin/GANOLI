config = {
    'architecture': {
        'generator': 'linear',
        'discriminator': 'linear',
    },
    'training': {
        'generator_lr': 1e-3,
        'discriminator_lr': 1e-3,
        'generator_loss': 'mse',
        'discriminator_loss': 'bce'
    },
}