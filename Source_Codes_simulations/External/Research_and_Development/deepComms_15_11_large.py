# Deep Learning Applications in Digital Communications
# By Rishabh Pomaje
#-----------------------------------------------------------------------------------
# >>> Abstract  
# DeepComms is experimentation, fusion of Deep Learning and Neural Networks and 
# their applications in the realm of digital communication systems.
# In this script we will be trying to learn (15, 11) Channel codes 
#-----------------------------------------------------------------------------------
# Importing the necessary dependencies :
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
print(tf.__version__)
TF_ENABLE_ONEDNN_OPTS = 0
#-----------------------------------------------------------------------------------
# System Specification Parameters/ Constants :
M = 2 ** 11                                        # Size of alphabet
k = np.log2(M)                                      # Number of bits required
n = 15                                              # Size of coded vector 
R = k / n                                           # Communication rate 
#-----------------------------------------------------------------------------------
# Creating the data to be trained on :
# 2^k dimensional One-Hot Encoded vectors will be used
data_size = 10 ** 5 # in number of blocks of 11 bits
M_indices = np.arange(0, M)
sample_indices = np.random.randint(0, M, data_size)

# Set of One Hot Encoded Vectors :
x_train = []
for idx in sample_indices:
    temp = np.zeros(M)
    temp[idx] = 1
    x_train.append(temp)
    
x_train = tf.constant(x_train)
# Labels for the data :
# Since we want to reproduce the input at the output :
y_train = x_train
#-----------------------------------------------------------------------------------
# Creating the Auto-Encoder Model : 
# Describing the encoder layers :
enc_input_layer = tf.keras.Input(shape=(M,), name='Input Layer')
enc_layer_01 = tf.keras.layers.Dense(M, activation='relu', name='Encoder_Hidden_01')(enc_input_layer)
enc_layer_02 = tf.keras.layers.Dense(n, activation='linear', name='Encoder_Hidden_02')(enc_layer_01)
enc_layer_normalized = tf.keras.layers.Lambda((lambda x: np.sqrt(n) * tf.keras.backend.l2_normalize(x, axis=-1)))(enc_layer_02)

# Describing the channel layers :
# Training SNR
SNR_dB = 7                                       # Eb / N0 in dB scale
SNR_lin = 10 ** (SNR_dB / 10)                    # In linear scale
ch_noise_layer = tf.keras.layers.GaussianNoise(stddev=np.sqrt(1 / (2 * R * SNR_lin)), name='AWGN_channel')(enc_layer_normalized)

# Describing the decoder layers :
dec_layer_01 =  tf.keras.layers.Dense(4*M, activation='relu', name='Decoder_Hidden_01')(ch_noise_layer)
dec_layer_02 =  tf.keras.layers.Dense(2*M, activation='relu', name='Decoder_Hidden_02')(dec_layer_01)
dec_output_layer = tf.keras.layers.Dense(M, activation='softmax', name='Output_Layer')(dec_layer_02)

autoencoder = tf.keras.Model(enc_input_layer,dec_output_layer)
#-----------------------------------------------------------------------------------
# Compiling the model :
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
autoencoder.summary()
#-----------------------------------------------------------------------------------
# Fitting the model by using training set :
history = autoencoder.fit(x_train, y_train, batch_size=1000, epochs=250) 
#-----------------------------------------------------------------------------------
# Display the results/ history :
fig, axes = plt.subplots()
axes.plot(history.history['accuracy'], label='Accuracy', color='blue')
axes.plot(history.history['loss'], label='Loss', color='red')
axes.set_title('Model History')
axes.set_ylabel('accuracy/ loss')
axes.set_xlabel('# epochs')
axes.grid()
axes.legend()
#-----------------------------------------------------------------------------------
# Testing the above learned system
# Creating the individual Encoder/ Decoder Models for use

# Encoder :
encoder_model = tf.keras.Model(enc_input_layer, enc_layer_normalized)

# Supposed received codeword at the receiver
encoded_input = tf.keras.Input(shape=(n,))
decoder_output = autoencoder.layers[-3](encoded_input)
decoder_output = autoencoder.layers[-2](decoder_output)
decoder_output = autoencoder.layers[-1](decoder_output)
# Decoder :
decoder_model = tf.keras.Model(encoded_input, decoder_output)
#-----------------------------------------------------------------------------------
# Creating test data
test_data_size = 10 ** 6 # Number of blocks
test_labels = np.random.randint(0, M, test_data_size)
test_vectors = []
for idx in test_labels:
    temp = np.zeros(M)
    temp[idx] = 1
    test_vectors.append(temp)

test_vectors = tf.constant(test_vectors)
#-----------------------------------------------------------------------------------
# Range of Signal to Noise Ratio :
# in dB :
SNR_dB = np.linspace(-4, 8, 25)
# in Linear Scale :
SNR_lin = 10 ** (SNR_dB / 10)
# Fixing energy per bit :
E_b = 1 
# Range of noise variance accordingly :
noise_var = 1 / (2 * R * SNR_lin) 

# Testing the learned system on validation data
BLER_learned = []
for noise in noise_var :
    # Encoding using our model :
    encoded_signal = encoder_model.predict(test_vectors)
    # Generating AWGN samples :
    awgn = np.sqrt(noise) * np.random.normal(0, 1, (test_data_size, n))
    rx_noisy_signal = encoded_signal + awgn
    # Decoding using our model :
    decoded_signal = decoder_model.predict(rx_noisy_signal)
    estimated_vectors = np.argmax(decoded_signal, axis=-1)
    BLER_learned.append(np.sum(estimated_vectors != test_labels) / test_data_size)
#-----------------------------------------------------------------------------------
BLER_uncoded = [0.8963204963204964, 0.8747648747648747, 0.8501094501094502, 0.8204534204534204, 0.7857857857857858, 0.746027346027346, 0.7008261008261009, 0.6492096492096492, 0.5942183942183942, 0.5344883344883344, 0.4723844723844724, 0.4072996072996073, 0.34344014344014345, 0.28166188166188166, 0.22516802516802517, 0.17416097416097417, 0.12928092928092927, 0.09246829246829247, 0.0631928631928632, 0.041707641707641706, 0.025678425678425678, 0.015353815353815354, 0.008518408518408519, 0.004428604428604429, 0.0019998019998019997]
BLER_coded_hard = [0.8788502788502789, 0.8528990528990529, 0.8223102223102223, 0.7868505868505868, 0.743932943932944, 0.696980496980497, 0.641967241967242, 0.5832205832205832, 0.5182325182325183, 0.4499246499246499, 0.3801075801075801, 0.31112651112651113, 0.2472912472912473, 0.18811338811338812, 0.13742093742093742, 0.0948068948068948, 0.06273746273746274, 0.038533038533038536, 0.02244002244002244, 0.012238612238612239, 0.005966405966405967, 0.0028732028732028733, 0.001166001166001166, 0.000429000429000429, 0.0001254001254001254]
BLER_coded_MLD = [0.835094435094435, 0.8003080003080003, 0.7588445588445588, 0.7105149105149106, 0.6573540573540574, 0.5954195954195954, 0.5285307285307286, 0.455946055946056, 0.38364298364298366, 0.3118371118371118, 0.24306284306284306, 0.18166298166298167, 0.1287023287023287, 0.08510268510268511, 0.05355685355685356, 0.03136103136103136, 0.016436216436216435, 0.00841060841060841, 0.0036586036586036584, 0.0014212014212014212, 0.0005588005588005588, 0.0001914001914001914, 4.4000044000044e-05, 1.32000132000132e-05, 0.0]
#-----------------------------------------------------------------------------------
# Visualizing the results
plt.rcParams.update({
    "text.usetex": True,
    "font.family" : 'serif',
    "font.size": 16
})
fig, axes = plt.subplots()
axes.semilogy(SNR_dB, BLER_uncoded, label="Uncoded", c='black')
axes.semilogy(SNR_dB, BLER_coded_hard, label="(15, 11) Hamming Hard Decision", c="blue", ls="-.")
axes.semilogy(SNR_dB, BLER_learned, label="Learned System", c='red', marker='o', ls=":")
axes.semilogy(SNR_dB, BLER_coded_MLD, label="(15, 11) Hamming MLD", c="blue", ls="--")
axes.set_xlabel(r'$E_b / N_0[dB]$')
axes.set_ylabel(r'$P_{BLE}$')
axes.set_xlim(-4, 8)
axes.set_ylim(10**-5, 10**0)
axes.set_title(f'BLER Performance Comparison - BPSK')
axes.legend()
axes.grid()
#-----------------------------------------------------------------------------------
fig.savefig(fname='plots/deepComms_15_11_large_t.svg', transparent=True)

print(BLER_learned) # Learned schemes block error rate

with open("results/results_deepComms_15_11_large.txt", mode='w') as file_id :
    file_id.write("BLER_uncoded :\n")
    file_id.write(f'{str(BLER_uncoded)}\n')
    file_id.write("BLER_coded_hard :\n")
    file_id.write(f'{str(BLER_coded_hard)}\n')
    file_id.write("BLER_coded_MLD :\n")
    file_id.write(f'{str(BLER_coded_MLD)}\n')
    file_id.write("BLER_learned :\n")
    file_id.write(str(BLER_learned))




