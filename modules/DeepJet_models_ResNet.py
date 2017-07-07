import keras

kernel_initializer = 'he_normal'
bn_axis = -1

def FC(data, num_hidden, act='relu', p=None, name=''):
    fc = keras.layers.Dense(num_hidden, activation=act, name='%s_relu' % name)(data)
    if not p:
        return fc
    else:
        dropout = keras.layers.Dropout(rate=p, name='%s_dropout' % name)(fc)
        return dropout

def residual_unit(data, num_filter, stride, dim_match, name, height=1, bottle_neck=True, bn_mom=0.9):
    if bottle_neck:
        bn1 = keras.layers.BatchNormalization(axis=bn_axis, momentum=bn_mom, epsilon=2e-5, name='%s_bn1' % name)(data)
        act1 = keras.layers.Activation('relu', name='%s_relu1' % name)(bn1)
        conv1 = keras.layers.Conv1D(filters=int(num_filter * 0.25), kernel_size=(1,), strides=(1,), padding='same',
                                    kernel_initializer=kernel_initializer, use_bias=False, name='%s_conv1' % name)(act1)

        bn2 = keras.layers.BatchNormalization(axis=bn_axis, momentum=bn_mom, epsilon=2e-5, name='%s_bn2' % name)(conv1)
        act2 = keras.layers.Activation('relu', name='%s_relu2' % name)(bn2)
        conv2 = keras.layers.Conv1D(filters=int(num_filter * 0.25), kernel_size=(3,), strides=stride, padding='same',
                                    kernel_initializer=kernel_initializer, use_bias=False, name='%s_conv2' % name)(act2)

        bn3 = keras.layers.BatchNormalization(axis=bn_axis, momentum=bn_mom, epsilon=2e-5, name='%s_bn3' % name)(conv2)
        act3 = keras.layers.Activation('relu', name='%s_relu3' % name)(bn3)
        conv3 = keras.layers.Conv1D(filters=num_filter, kernel_size=(1,), strides=(1,), padding='same',
                                    kernel_initializer=kernel_initializer, use_bias=False, name='%s_conv3' % name)(act3)
        if dim_match:
            shortcut = data
        else:
            shortcut = keras.layers.Conv1D(filters=num_filter, kernel_size=(1,), strides=stride, padding='same',
                                    kernel_initializer=kernel_initializer, use_bias=False, name='%s_shortcut' % name)(act1)
        return keras.layers.add([conv3, shortcut])

    else:
        bn1 = keras.layers.BatchNormalization(axis=bn_axis, momentum=bn_mom, epsilon=2e-5, name='%s_bn1' % name)(data)
        act1 = keras.layers.Activation('relu', name='%s_relu1' % name)(bn1)
        conv1 = keras.layers.Conv1D(filters=num_filter, kernel_size=(3,), strides=stride, padding='same',
                                    kernel_initializer=kernel_initializer, use_bias=False, name='%s_conv1' % name)(act1)
        bn2 = keras.layers.BatchNormalization(axis=bn_axis, momentum=bn_mom, epsilon=2e-5, name='%s_bn2' % name)(conv1)
        act2 = keras.layers.Activation('relu', name='%s_relu2' % name)(bn2)
        conv2 = keras.layers.Conv1D(filters=num_filter, kernel_size=(3,), strides=(1,), padding='same',
                                    kernel_initializer=kernel_initializer, use_bias=False, name='%s_conv2' % name)(act2)
        if dim_match:
            shortcut = data
        else:
            shortcut = keras.layers.Conv1D(filters=num_filter, kernel_size=(1,), strides=stride, padding='same',
                                    kernel_initializer=kernel_initializer, use_bias=False, name='%s_shortcut' % name)(act1)
        return keras.layers.add([conv2, shortcut])


def resnet(data, units, filter_list, num_classes, height=1, bottle_neck=True, bn_mom=0.9, name=''):
    num_stages = len(units)
    data = keras.layers.BatchNormalization(axis=bn_axis, momentum=bn_mom, epsilon=2e-5, scale=False, name='%s_bn_data' % name)(data)
    body = keras.layers.Conv1D(filters=filter_list[0], kernel_size=(3,), strides=(1,), padding='same',
                                    kernel_initializer=kernel_initializer, use_bias=False, name='%s_conv0' % name)(data)

    for i in range(num_stages):
        body = residual_unit(body, filter_list[i + 1], stride=(1 if i == 0 else 2,), dim_match=False,
                             height=height, name='%s_stage%d_unit%d' % (name, i + 1, 1), bottle_neck=bottle_neck)
        for j in range(units[i] - 1):
            body = residual_unit(body, filter_list[i + 1], stride=(1,), dim_match=True, height=height,
                                 name='%s_stage%d_unit%d' % (name, i + 1, j + 2), bottle_neck=bottle_neck)

    bn1 = keras.layers.BatchNormalization(axis=bn_axis, momentum=bn_mom, epsilon=2e-5, name='%s_bn1' % name)(body)
    act1 = keras.layers.Activation('relu', name='%s_relu1' % name)(bn1)
    pool = keras.layers.GlobalAveragePooling1D(name='%s_pool' % name)(act1)
#     flat = keras.layers.Flatten(name='%s_flatten' % name)(pool)
    return pool

def get_subnet(num_classes, input_name, data, height=1, filter_list=[64, 128, 256, 512, 1024], bottle_neck=True, units=[3, 4, 6, 3]):
    return resnet(data,
                  units=units,
                  name=input_name,
                  filter_list=filter_list,
                  height=height,
                  num_classes=num_classes,
                  bottle_neck=bottle_neck)

def resnet_model(inputs, num_classes,num_regclasses, **kwargs):

    input_jet = inputs[0]
    input_cpf = inputs[1]
    input_npf = inputs[2]
    input_sv = inputs[3]
    
    input_regDummy=inputs[4]
    
    reg=keras.layers.Dense(2,kernel_initializer='ones',trainable=False,name='reg_off')(input_regDummy)

    cpf = get_subnet(num_classes, data=input_cpf, input_name='Cpfcan', filter_list=[32, 64, 64, 128], bottle_neck=False, units=[2, 2, 2])
    npf = get_subnet(num_classes, data=input_npf, input_name='Npfcan', filter_list=[32, 32, 64, 64], bottle_neck=False, units=[2, 2, 2])
    sv = get_subnet(num_classes, data=input_sv, input_name='sv', filter_list=[32, 32, 64], bottle_neck=False, units=[3, 3])

    concat = keras.layers.concatenate([input_jet, cpf, npf, sv], name='concat')
    fc1 = FC(concat, 512, p=0.2, name='fc1')
    output = keras.layers.Dense(num_classes, activation='softmax', name='softmax')(fc1)

    model = keras.models.Model(inputs=inputs, outputs=[output,reg])

    return model

