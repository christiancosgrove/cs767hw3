# CS767 HW3

## Training HRED model
Christian Cosgrove, Darius Irani, Jung Min Lee



### How we added HRED

We used [Harshal's HRED implementation](https://github.com/hsgodhia/hred); with minor modifications, we were able to integrate it with ParlAI. We also implemented a ParlAI Task for the MovieTriples dataset (data provided by Joao). 

What we modified:

1. MovieTriples Task -- `/parlai/tasks/movie_triples/agents.py`. We created a dataloader where each triple is a single utterance. We experimented with different approaches here--e.g. considering each utterance separately and using the [history vector functionality](https://github.com/facebookresearch/ParlAI/issues/2404). However, the simplest approach we decided on was to (at training time) consider the triple to be a single utterance separated by `</s>` tokens and split these utterances in the `batchify` method. (At inference time, we used the `history_vecs` instead).

1. Dictionary creation. Part of the challenge of working with the MovieTriples data is that it has custom separation/person tokens. Rather than hardcoding this, we wanted to use the most ParlAI-friendly code. Noticing that `train_model.py` will load a `.dict` file if it is already present, we wrote code to generate a ParlAI dictionary directly from the dictionary provided with the MovieTriples data. Run `create_dict_pickle.py` to generate a ParlAI `.dict` before running `train_model.py` or `interactive.py` to ensure that tokens are mapped correctly.

2. `HredAgent` -- This is the core ParlAI code for the HRED model. It inherits from `TorchGeneratorAgent`, which provides scaffolding code for an encoder-decoder-type model. However, we were unable to rely on most of the initial code because it was designed for a single encoder and decoder; in our case, we have two encoders (utterance encoder and session encoder), the first of which is called multiple times (on each of the input utterances). Therefore, we had to override the `_generate` and `_compute_loss` methods in `HredAgent`, calling Harshal's code instead.

3. Alexa integration. TODO