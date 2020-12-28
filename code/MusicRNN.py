import tensorflow as tf
import numpy as np
import os
from datetime import datetime
import time
import TextToAudio
import random
import progressbar

import pretty_midi
import string

# Load data from dir
data_dir = './data/kirby_data.txt'
checkpoint_dir = './training_checkpoints'

train = True
loadData = True

# Read, then decode for py2 compat.
text = open(data_dir, 'rb').read().decode(encoding='utf-8')

# Get all unique characters in the file
vocab = sorted(set(text))

# Create a dict to translate to and from indexed characters to real characters
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

# Apply char to int translation
text_as_int = np.array([char2idx[c] for c in text])

# The maximum length sentence we want for a single input in characters
seq_length = 400 # 1000
examples_per_epoch = len(text)//(seq_length+1)

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

# Split up dataset in sentence sized batches
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

# For each sequence, copy and shift it to use as prediction data
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

# Batch size
BATCH_SIZE = 32 # 64  
# Buffer size (for shuffling)
BUFFER_SIZE = 10000 # 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
print(dataset)
EPOCHS = 2000
# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256 # 256

# Number of RNN units (neurons?)
rnn_units = 512 # 1024

# Build the GRU 
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

# Build model
model = build_model(vocab_size = len(vocab), embedding_dim=embedding_dim, rnn_units=rnn_units, batch_size=BATCH_SIZE)
if loadData:
  try:
    model.load_weights(checkpoint_dir+'/_model.h5')
    print('LOADED WEIGHTS FROM _MODEL.H5')
  except:
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    print('LOADED WEIGHTS FROM CKPT')
    
# Display the prediction shape
for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

model.summary()

# soup time
sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
print()
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices])))

#with open(f"./generated/not_trained.txt", "w") as file:
#    file.seek(0)
#    file.truncate()
#    file.writelines("".join(idx2char[sampled_indices]))

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss  = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())
optimizer = tf.keras.optimizers.Adam(1e-4, 0.5)
model.compile(optimizer=optimizer, loss=loss)
# Directory where the checkpoints will be saved
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

@tf.function
def train_step(inp, target):
  with tf.GradientTape() as tape:
    predictions = model(inp)
    loss = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            target, predictions, from_logits=True))
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  return loss

SUFFIXES = {1: 'st', 2: 'nd', 3: 'rd'}
ACCIDENTALS = ['#', 'b', '', '', '', '', '']
def ordinal(num):
    if 10 <= num % 100 <= 20:
        suffix = 'th'
    else:
        suffix = SUFFIXES.get(num % 10, 'th')
    return str(num) + suffix
def random_letter():
  return random.choice(string.ascii_letters).upper() + random.choice(ACCIDENTALS)
# Generate 
random.seed(datetime.now())

def generate(epoch, loss):
  epoch = epoch
  gmodel = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
  try:
    gmodel.load_weights(checkpoint_dir+'/_model.h5')
    print('GEN LOADED WEIGHTS FROM _MODEL.H5')
  except:
    print('Cannot load weights')
  print('Building RNN for generation')
  gmodel.build(tf.TensorShape([1, None]))
  print('Done')

  def generate_text(gmodel, start_string):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = 10000

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1

    # Here batch size == 1
    gmodel.reset_states()
    for i in range(num_generate):
        predictions = gmodel(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # We pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (''.join(text_generated)) # ''.join(start_string) + 

  text_gen = ''
  names = ['Symphony', 'Suite','Variations','Rondo','Requiem','Rhapsody','Sonata','Concerto','Allemande','Courante','Sarabande','Gigue','Minuet','Gavotte','Bourree','Waltz','Tango', 'Overture','Prelude','Intermezzo','Interlude','Finale','Presto','Nocturne','Impromptu','Divertimento','Ballade','Berceuse', 'Palazzolo', 'Modica', 'Noto', 'Ispica', 'Toccata', 'Fugue', 'Puccini', 'Carmenda', 'Bolero', 'Larghetto', 'Inventions', 'Theme', 'Canon', 'Ode', 'Joy', 'A Simple Song', 'Ciacona', 'Fantasia', 'Praeludium']
  markers = [f'in {random_letter()} Major',f'in {random_letter()} Minor', f'(Op.{epoch})', f'No.{epoch}', f'{ordinal(epoch)} Movement', f'Variation {epoch}']
  filename = str(epoch) + "_" + random.choice(names) + ' ' + random.choice(markers)
  print(f'{filename}')
  with open(f"./generated/{filename}.txt", "w") as file:
    file.seek(0)
    file.truncate()
  print(f'Created and wiped file {filename}.txt')
  index = random.randint(0, len(text) - 1000)
  start_string = text[index:index + 990]
  print(f'Started text generation, Prompt: \n{start_string}')
  print('Generating...')
  text_gen = ''
  text_gen = generate_text(gmodel, start_string=start_string)
  print(f'Done')
  with open(f"./generated/{filename}_info.txt", "w") as file:
    file.seek(0)
    file.truncate()
    file.write(f'Start: \n{start_string}, Loss:{loss}')
  with open(f"./generated/{filename}.txt", "a+") as file:
    #file.write(start_string.replace('\n', ''))
    #file.write(' ' * 10)
    file.write(text_gen.replace('\n', ''))
  print('Decoding generated text')
  TextToAudio.generate_audio(filename)
  print('Done')

# Training step
batch_size = 0
for epoch in range(EPOCHS):
  start = time.time()

  # initializing the hidden state at the start of every epoch
  # initally hidden is None
  hidden = model.reset_states()
  b = 0
  #bar = progressbar.ProgressBar(max_value=(batch_size + 2) if epoch != 0 else 100, widgets=[f'Epoch {epoch+1}', progressbar.Bar(), '(', progressbar.ETA(),')'])
  for (batch_n, (inp, target)) in enumerate(dataset):
    loss = train_step(inp, target)

    if batch_n > batch_size:
      batch_size = batch_n
    if batch_n % 2 == 0:
      template = 'Epoch {} Batch {}/{} Loss {}'
      print(template.format(epoch+1, batch_n, batch_size, loss))
    b += 1
    #bar.update(b)

  model.save_weights(checkpoint_dir+'/_model.h5')#checkpoint_prefix.format(epoch=epoch))
  generate(epoch+1, loss)

  print ('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
  print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
model.save_weights(checkpoint_prefix.format(epoch=epoch))
print('Done')