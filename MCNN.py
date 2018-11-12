from keras.layers import Conv2D, MaxPooling2D, Concatenate, Input
from keras.models import Model


def MCNN_body_branch(input_flow, flow_mode='large'):
    if flow_mode == 'large':
        filter_num_initial, conv_len_initial, maxpooling_size = 16, 9, (2, 2)
    elif flow_mode == 'medium':
        filter_num_initial, conv_len_initial, maxpooling_size = 20, 7, (2, 2)
    elif flow_mode == 'small':
        filter_num_initial, conv_len_initial, maxpooling_size = 24, 5, (2, 2)
    else:
        print('Only small/medium/large modes.')
        return None
    x = Conv2D(filter_num_initial, (conv_len_initial, conv_len_initial), padding='same', activation='relu')(input_flow)
    x = MaxPooling2D(pool_size=maxpooling_size)(x)
    x = (x)
    x = Conv2D(filter_num_initial*2, (conv_len_initial-2, conv_len_initial-2), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=maxpooling_size)(x)
    x = Conv2D(filter_num_initial, (conv_len_initial-2, conv_len_initial-2), padding='same', activation='relu')(x)
    x = Conv2D(filter_num_initial//2, (conv_len_initial-2, conv_len_initial-2), padding='same', activation='relu')(x)
    return x


def MCNN(weights=None, input_shape=(None, None, 1)):
    input_flow = Input(shape=input_shape)
    branches = []
    for flow_mode in ['large', 'medium', 'small']:
        branches.append(MCNN_body_branch(input_flow, flow_mode=flow_mode))
    merged_feature_maps = Concatenate(axis=3)(branches)
    density_map = Conv2D(1, (1, 1), padding='same')(merged_feature_maps)

    model = Model(inputs=input_flow, outputs=density_map)
    
    return model
