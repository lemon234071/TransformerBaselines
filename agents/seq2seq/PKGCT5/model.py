import torch
from torch.nn import CrossEntropyLoss
from transformers import T5PreTrainedModel, MT5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import (add_start_docstrings_to_model_forward, T5_INPUTS_DOCSTRING,
                                                replace_return_docstrings, Seq2SeqLMOutput, _CONFIG_FOR_DOC,
                                                BaseModelOutput)
from transformers.generation_utils import ModelOutput, Any, Dict

from modules.MayiGCN import MayiGCN


class PKGCT5mayi(MT5ForConditionalGeneration):
    def __init__(self, config, gcn_num_layer=2):
        super(PKGCT5mayi, self).__init__(config)
        self.spell_gcn = MayiGCN(gcn_num_layer, config.d_model, dropout_rate=0.9)

    def _prepare_graph_embeds(self, input_ids, adj, n2w, w2n):
        inputs_embeds = self.shared(input_ids)

        ## get vocab emb
        gcn_in = self.shared(n2w)
        ## run gcn on whole graph
        graph_embeds = self.spell_gcn(gcn_in, adj)
        ## convert (temp)
        # graph_embeds = torch.gather(graph_embeds, 0, w2n)
        extend_graph_embeds = graph_embeds[w2n]
        ## mask pad
        pad_mask = w2n.ne(0).unsqueeze(1)
        # graph_embeds[pad_mask] = graph_embeds[pad_mask] * 0
        masked_extend_graph_embeds = pad_mask.float() * extend_graph_embeds
        ## extract batch nodes emb ( flat then gather)
        flat_inputs_idx = torch.flatten(input_ids)
        # temp_map = torch.gather(w2n, 1, flat_inputs_idx)
        # inputs_graph_embeds = torch.gather(graph_embeds, 0, temp_map)
        inputs_graph_embeds = masked_extend_graph_embeds[flat_inputs_idx]
        inputs_graph_embeds = inputs_graph_embeds.view(inputs_embeds.shape)
        inputs_embeds = inputs_embeds + inputs_graph_embeds
        return inputs_embeds

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            head_mask=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            adj=None,
            n2w=None,
            w2n=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``

        Returns:

        Examples::

            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')

            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits

            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        ### TODO(yida)
        if adj is not None:
            inputs_embeds = self._prepare_graph_embeds(input_ids, adj, n2w=n2w, w2n=w2n)
            ## None the inputs_ids
            input_ids = None
        ###

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def _prepare_encoder_decoder_kwargs_for_generation(
            self, input_ids: torch.LongTensor, model_kwargs
    ) -> Dict[str, Any]:

        ### TODO(yida)
        adj, w2n, n2w = model_kwargs.pop("adj"), model_kwargs.pop("w2n"), model_kwargs.pop("n2w")
        inputs_embeds = self._prepare_graph_embeds(input_ids, adj, n2w=n2w, w2n=w2n)

        # retrieve encoder hidden states
        encoder = self.get_encoder()
        encoder_kwargs = {
            argument: value for argument, value in model_kwargs.items() if not argument.startswith("decoder_")
        }
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(None, inputs_embeds=inputs_embeds, return_dict=True,
                                                               **encoder_kwargs)
        return model_kwargs
