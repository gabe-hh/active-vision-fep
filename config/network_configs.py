# Encoder and Decoder Configurations
enc_3layer_16to64 =[
        {'filters': 16, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1},
        {'filters': 32, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1},
        {'filters': 64, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1}
    ]
enc_3layer_16to64_reducedfilm = [
        {'filters': 16, 'kernel_size': 3, 'stride': 2, 'film': False, 'padding': 1},
        {'filters': 32, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1},
        {'filters': 64, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1}
    ]
enc_3layer_8to32 = [
        {'filters': 8, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1},
        {'filters': 16, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1},
        {'filters': 32, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1}
    ]
enc_3layer_8to32_nofilm = [
        {'filters': 8, 'kernel_size': 3, 'stride': 2, 'film': False, 'padding': 1},
        {'filters': 16, 'kernel_size': 3, 'stride': 2, 'film': False, 'padding': 1},
        {'filters': 32, 'kernel_size': 3, 'stride': 2, 'film': False, 'padding': 1}
    ]
enc_3layer_32to128 = [
        {'filters': 32, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1},
        {'filters': 64, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1},
        {'filters': 128, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1}
    ]
enc_4layer_8to64 = [
        {'filters': 8, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1},
        {'filters': 16, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1},
        {'filters': 32, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1},
        {'filters': 64, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1}
    ]
enc_4layer_8to64_reducedfilm = [
        {'filters': 8, 'kernel_size': 3, 'stride': 2, 'film': False, 'padding': 1},
        {'filters': 16, 'kernel_size': 3, 'stride': 2, 'film': False, 'padding': 1},
        {'filters': 32, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1},
        {'filters': 64, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1}
    ]
enc_5layer_8to128 = [
        {'filters': 8, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1},
        {'filters': 16, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1},
        {'filters': 32, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1},
        {'filters': 64, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1},
        {'filters': 128, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1}
    ]
enc_5layer_8to128_reducedfilm = [
        {'filters': 8, 'kernel_size': 3, 'stride': 2, 'film': False, 'padding': 1},
        {'filters': 16, 'kernel_size': 3, 'stride': 2, 'film': False, 'padding': 1},
        {'filters': 32, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1},
        {'filters': 64, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1},
        {'filters': 128, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1}
    ]

enc_ssm_3layer_8to32 = [
        {'filters': 8, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1},
        {'filters': 16, 'kernel_size': 3, 'stride': 1, 'film': True, 'padding': 1},
        {'filters': 32, 'kernel_size': 3, 'stride': 1, 'film': True}
    ]
enc_ssm_3layer_16to64 = [
        {'filters': 16, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1},
        {'filters': 32, 'kernel_size': 3, 'stride': 1, 'film': True},
        {'filters': 64, 'kernel_size': 3, 'stride': 1, 'film': True}
    ]
enc_ssm_4layer_8to64 = [
        {'filters': 8, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1},
        {'filters': 16, 'kernel_size': 3, 'stride': 1, 'film': True},
        {'filters': 32, 'kernel_size': 3, 'stride': 1, 'film': True},
        {'filters': 64, 'kernel_size': 3, 'stride': 1, 'film': True}
    ]

enc_ssm_2layer_16to32_nofilm = [
    {'filters': 16, 'kernel_size': 3, 'stride': 2, 'film': False, 'padding': 1},
    {'filters': 32, 'kernel_size': 3, 'stride': 1, 'film': False},
]

enc_ssm_3layer_8to32_nofilm = [
    {'filters': 8, 'kernel_size': 3, 'stride': 2, 'film': False, 'padding': 1},
    {'filters': 16, 'kernel_size': 3, 'stride': 1, 'film': False, 'padding': 1},
    {'filters': 32, 'kernel_size': 3, 'stride': 1, 'film': False},
]

enc_ssm_3layer_16to64_nofilm = [
    {'filters': 16, 'kernel_size': 3, 'stride': 2, 'film': False, 'padding': 1},
    {'filters': 32, 'kernel_size': 3, 'stride': 1, 'film': False, 'padding': 1},
    {'filters': 64, 'kernel_size': 3, 'stride': 1, 'film': False},
]

dec_invssm_2layer_32to16_nofilm = [
    {'filters': 32, 'kernel_size': 3, 'stride': 1, 'film': False},
    {'filters': 16, 'kernel_size': 3, 'stride': 2, 'film': False, 'padding': 1, 'output_padding': 1},
]

enc_ssm_2layer_16to32 = [
    {'filters': 16, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1},
    {'filters': 32, 'kernel_size': 3, 'stride': 1, 'film': True},
]

dec_invssm_2layer_32to16 = [
    {'filters': 32, 'kernel_size': 3, 'stride': 1, 'film': True},
    {'filters': 16, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1, 'output_padding': 1},
]

dec_invssm_3layer_32to8_nofilm = [
    {'filters': 32, 'kernel_size': 3, 'stride': 1, 'film': False},
    {'filters': 16, 'kernel_size': 3, 'stride': 1, 'film': False, 'padding': 1},
    {'filters': 8, 'kernel_size': 3, 'stride': 2, 'film': False, 'padding': 1, 'output_padding': 1},
]

dec_invssm_3layer_64to16_nofilm = [
    {'filters': 64, 'kernel_size': 3, 'stride': 1, 'film': False},
    {'filters': 32, 'kernel_size': 3, 'stride': 1, 'film': False, 'padding': 1},
    {'filters': 16, 'kernel_size': 3, 'stride': 2, 'film': False, 'padding': 1, 'output_padding': 1},
]

dec_invssm_4layer_64to8_nofilm = [
    {'filters': 64, 'kernel_size': 3, 'stride': 1, 'film': False},
    {'filters': 32, 'kernel_size': 3, 'stride': 1, 'film': False},
    {'filters': 16, 'kernel_size': 3, 'stride': 1, 'film': False, 'padding': 1},
    {'filters': 8, 'kernel_size': 3, 'stride': 2, 'film': False, 'padding': 1, 'output_padding': 1},
]

dec_invssm_3layer_32to8 = [
    {'filters': 32, 'kernel_size': 3, 'stride': 1, 'film': True},
    {'filters': 16, 'kernel_size': 3, 'stride': 1, 'film': True, 'padding': 1},
    {'filters': 8, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1, 'output_padding': 1},
]

dec_3layer_64to16 = [
        {'filters': 64, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1, 'output_padding': 1},
        {'filters': 32, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1, 'output_padding': 1},
        {'filters': 16, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1, 'output_padding': 1}
    ]
dec_3layer_32to8 = [
        {'filters': 32, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1, 'output_padding': 1},
        {'filters': 16, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1, 'output_padding': 1},
        {'filters': 8, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1, 'output_padding': 1}
    ]
dec_3layer_32to8_nofilm = [
        {'filters': 32, 'kernel_size': 3, 'stride': 2, 'film': False, 'padding': 1, 'output_padding': 1},
        {'filters': 16, 'kernel_size': 3, 'stride': 2, 'film': False, 'padding': 1, 'output_padding': 1},
        {'filters': 8, 'kernel_size': 3, 'stride': 2, 'film': False, 'padding': 1, 'output_padding': 1}
    ]
dec_3layer_32to8_reducedfilm = [
        {'filters': 32, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1, 'output_padding': 1},
        {'filters': 16, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1, 'output_padding': 1},
        {'filters': 8, 'kernel_size': 3, 'stride': 2, 'film': False, 'padding': 1, 'output_padding': 1}
    ]
dec_3layer_32to3 = [
        {'filters': 32, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1, 'output_padding': 1},
        {'filters': 16, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1, 'output_padding': 1},
        {'filters': 3, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1, 'output_padding': 1}
    ]
dec_4layer_64to8 = [
        {'filters': 64, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1, 'output_padding': 1},
        {'filters': 32, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1, 'output_padding': 1},
        {'filters': 16, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1, 'output_padding': 1},
        {'filters': 8, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1, 'output_padding': 1}
    ]
dec_4layer_64to8_reducedfilm = [
        {'filters': 64, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1, 'output_padding': 1},
        {'filters': 32, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1, 'output_padding': 1},
        {'filters': 16, 'kernel_size': 3, 'stride': 2, 'film': False, 'padding': 1, 'output_padding': 1},
        {'filters': 8, 'kernel_size': 3, 'stride': 2, 'film': False, 'padding': 1, 'output_padding': 1}
    ]
dec_4layer_128to16 = [
        {'filters': 128, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1, 'output_padding': 1},
        {'filters': 64, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1, 'output_padding': 1},
        {'filters': 32, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1, 'output_padding': 1},
        {'filters': 16, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1, 'output_padding': 1}
    ]
dec_upscale_3layer_64to16 = [
        {'type': 'upscale_conv', 'filters': 64, 'kernel_size': 3, 'stride': 1, 'film': True, 'padding': 1},
        {'type': 'upscale_conv', 'filters': 32, 'kernel_size': 3, 'stride': 1, 'film': True, 'padding': 1},
        {'type': 'upscale_conv', 'filters': 16, 'kernel_size': 3, 'stride': 1, 'film': True, 'padding': 1}
    ]
dec_upscale_3layer_32to8 = [
        {'type': 'upscale_conv', 'filters': 32, 'kernel_size': 3, 'stride': 1, 'film': True, 'padding': 1},
        {'type': 'upscale_conv', 'filters': 16, 'kernel_size': 3, 'stride': 1, 'film': True, 'padding': 1},
        {'type': 'upscale_conv', 'filters': 8, 'kernel_size': 3, 'stride': 1, 'film': True, 'padding': 1}
    ]
dec_upscale_4layer_64to8 = [
        {'type': 'upscale_conv', 'filters': 64, 'kernel_size': 3, 'stride': 1, 'film': True, 'padding': 1},
        {'type': 'upscale_conv', 'filters': 32, 'kernel_size': 3, 'stride': 1, 'film': True, 'padding': 1},
        {'type': 'upscale_conv', 'filters': 16, 'kernel_size': 3, 'stride': 1, 'film': True, 'padding': 1},
        {'type': 'upscale_conv', 'filters': 8, 'kernel_size': 3, 'stride': 1, 'film': True, 'padding': 1}
    ]
dec_upscale_4layer_128to16 = [
        {'type': 'upscale_conv', 'filters': 128, 'kernel_size': 3, 'stride': 1, 'film': True, 'padding': 1},
        {'type': 'upscale_conv', 'filters': 64, 'kernel_size': 3, 'stride': 1, 'film': True, 'padding': 1},
        {'type': 'upscale_conv', 'filters': 32, 'kernel_size': 3, 'stride': 1, 'film': True, 'padding': 1},
        {'type': 'upscale_conv', 'filters': 16, 'kernel_size': 3, 'stride': 1, 'film': True, 'padding': 1}
    ]
dec_5layer_128to8 = [
        {'filters': 128, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1, 'output_padding': 1},
        {'filters': 64, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1, 'output_padding': 1},
        {'filters': 32, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1, 'output_padding': 1},
        {'filters': 16, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1, 'output_padding': 1},
        {'filters': 8, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1, 'output_padding': 1}
    ]
dec_5layer_128to8_reducedfilm = [
        {'filters': 128, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1, 'output_padding': 1},
        {'filters': 64, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1, 'output_padding': 1},
        {'filters': 32, 'kernel_size': 3, 'stride': 2, 'film': True, 'padding': 1, 'output_padding': 1},
        {'filters': 16, 'kernel_size': 3, 'stride': 2, 'film': False, 'padding': 1, 'output_padding': 1},
        {'filters': 8, 'kernel_size': 3, 'stride': 2, 'film': False, 'padding': 1, 'output_padding': 1}
    ]

encoders = [enc_3layer_16to64, enc_3layer_16to64_reducedfilm, enc_3layer_8to32, enc_3layer_32to128, enc_4layer_8to64, enc_4layer_8to64_reducedfilm, enc_5layer_8to128, enc_5layer_8to128_reducedfilm]
decoders = [dec_3layer_64to16,dec_3layer_32to8,dec_3layer_32to8_reducedfilm,dec_3layer_32to3, dec_4layer_64to8, dec_4layer_128to16, dec_upscale_3layer_64to16, dec_upscale_3layer_32to8, dec_upscale_4layer_64to8, dec_upscale_4layer_128to16, dec_5layer_128to8, dec_5layer_128to8_reducedfilm]
