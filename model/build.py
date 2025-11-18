



AVAILABLE_MODELS = [
	'spine_encoder',
	'spine_transformer',
	'cnn3d',
]



def build_model(name, params=None):
    key = name.lower()
    if key not in AVAILABLE_MODELS:
        raise KeyError(f"Unknown model '{name}'")

    model = name
    params = params or {}
    if not isinstance(params, dict):
        raise TypeError('params must be a dict of constructor keyword arguments')

    if model == 'cnn3d':
        from CNNencoder import CNN3DFeatureExtractor
        return CNN3DFeatureExtractor(**params)

    elif model == 'spine_encoder':
        from SpineEncoder import SpineEncoder
        return SpineEncoder(**params)

    elif model == 'spine_transformer':
        from SpineTransformer import SpineDecoder
        return SpineDecoder(**params)




if __name__ == '__main__':

	print('Available models:', ', '.join(AVAILABLE_MODELS))

	m = build_model('spine_encoder', {'img_size': (32, 32, 32), 'patch_size': (8, 8, 8), 'embed_dim': 64, 'num_layers': 2,"num_heads":4, "mlp_dim":128})
	print('Built model:', type(m), 'params:', sum(p.numel() for p in m.parameters()))
