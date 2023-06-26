def gaudi_esmselfoutput_forward(self, hidden_states, input_tensor):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.dropout(hidden_states)
    hidden_states = hidden_states + input_tensor
    return hidden_states


def gaudi_esmoutput_forward(self, hidden_states, input_tensor):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.dropout(hidden_states)
    hidden_states = hidden_states + input_tensor
    return hidden_states
