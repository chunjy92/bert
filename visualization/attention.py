# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module for postprocessing and displaying transformer attentions.

This module is designed to be called from an ipython notebook.
"""

import json
import os

import IPython.display as display

import numpy as np

vis_html = """
  <span style="user-select:none">
    Layer: <select id="layer"></select>
    Attention: <select id="att_type">
      <option value="all">All</option>
      <option value="a">Sentence A self-attention</option>
      <option value="b">Sentence B self-attention</option>
      <option value="ab">Sentence A -> Sentence B</option>
      <option value="ba">Sentence B -> Sentence A</option>
    </select>
  </span>
  <div id='vis'></div>
"""

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
vis_js = open(os.path.join(__location__, 'attention.js')).read()

def show(tokens_a, tokens_b, attn):
  self_att = resize(attn)

  attention = _get_attention(tokens_a, tokens_b, self_att)
  att_json = json.dumps(attention)
  _show_attention(att_json)


def _show_attention(att_json):
  display.display(display.HTML(vis_html))
  display.display(display.Javascript('window.attention = %s' % att_json))
  display.display(display.Javascript(vis_js))


def resize(att_mat, max_length=None):
  """Normalize attention matrices and reshape as necessary."""
  for i, att in enumerate(att_mat):
    # Add extra batch dim for viz code to work.
    if att.ndim == 3:
      att = np.expand_dims(att, axis=0)
    if max_length is not None:
      # Sum across different attention values for each token.
      att = att[:, :, :max_length, :max_length]
      row_sums = np.sum(att, axis=2)
      # Normalize
      att /= row_sums[:, :, np.newaxis]
    att_mat[i] = att
  return att_mat


def _get_attention(tokens_a, tokens_b, attn):
  """Compute representation of the attention ready for the d3 visualization.

  Args:
    tokens_a: list of strings, words to be displayed on the left of the vis
    tokens_b: list of strings, words to be displayed on the right of the vis
    enc_atts: numpy array, encoder self-attentions
        [num_layers, batch_size, num_heads, enc_length, enc_length]
    dec_atts: numpy array, decoder self-attentions
        [num_layers, batch_size, num_heads, dec_length, dec_length]
    encdec_atts: numpy array, encoder-decoder attentions
        [num_layers, batch_size, num_heads, dec_length, enc_length]

  Returns:
    Dictionary of attention representations with the structure:
    {
      'all': Representations for showing all attentions at the same time.
      'inp_inp': Representations for showing encoder self-attentions
      'inp_out': Representations for showing encoder-decoder attentions
      'out_out': Representations for showing decoder self-attentions
    }
    and each sub-dictionary has structure:
    {
      'att': list of inter attentions matrices, one for each attention head
      'top_text': list of strings, words to be displayed on the left of the vis
      'bot_text': list of strings, words to be displayed on the right of the vis
    }
  """

  all_attns = []
  a_attns   = []
  b_attns   = []
  ab_attns  = []
  ba_attns  = []

  slice_a = slice(0, len(tokens_a))  # Positions corresponding to sentence A in input
  slice_b = slice(len(tokens_a), len(tokens_a) + len(tokens_b))  # Position corresponding to sentence B in input


  num_layers = len(attn)
  for layer in range(num_layers):
    layer_attn = attn[layer]  # Get layer attention (assume batch size = 1), shape = [num_heads, seq_len, seq_len]

    all_attns.append(layer_attn.tolist())  # Append AB->AB attention for layer, across all heads
    a_attns.append(layer_attn[slice_a, slice_a].tolist())  # Append A->A attention for layer, across all heads
    b_attns.append(layer_attn[slice_b, slice_b].tolist())  # Append B->B attention for layer, across all heads
    ab_attns.append(layer_attn[slice_a, slice_b].tolist())  # Append A->B attention for layer, across all heads
    ba_attns.append(layer_attn[slice_b, slice_a].tolist())  # Append B->A attention for layer, across all heads

  attentions = {
      'all': {
          'att'     : all_attns,
          'top_text': tokens_a + tokens_b,
          'bot_text': tokens_a + tokens_b,
      },
      'a': {
          'att'     : a_attns,
          'top_text': tokens_a,
          'bot_text': tokens_a,
      },
      'b': {
          'att'     : b_attns,
          'top_text': tokens_a,
          'bot_text': tokens_b,
      },
      'ab': {
          'att'     : ab_attns,
          'top_text': tokens_b,
          'bot_text': tokens_b,
      },
      'ba': {
        'att'     : ba_attns,
        'top_text': tokens_b,
        'bot_text': tokens_b,
      }
  }

  return attentions
