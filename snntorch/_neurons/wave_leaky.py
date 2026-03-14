from .neurons import SpikingNeuron
import torch
from torch import nn


class WaveLeaky(SpikingNeuron):
    """
    Klein-Gordon wave neuron model.
    Membrane potential follows second-order oscillatory dynamics driven
    by the Klein-Gordon dispersion relation instead of first-order
    exponential leaky integration.

    The update rule is:

    .. math::

            U[t+1] = 2 U[t] - U_{\\rm prev}[t] + \\Delta t^2
            (-\\chi^2 U[t] + I_{\\rm in}[t+1]) - R U_{\\rm thr}

    where :math:`\\chi` is the natural oscillation frequency
    (analogous to mass in the Klein-Gordon equation) and
    :math:`\\Delta t` is the integration time step.

    If `reset_mechanism = "subtract"`, then :math:`U[t+1]` will have
    `threshold` subtracted from it whenever the neuron emits a spike:

    .. math::

            U[t+1] = 2U[t] - U_{\\rm prev}[t] + \\Delta t^2
            (-\\chi^2 U[t] + I_{\\rm in}[t+1]) - R U_{\\rm thr}

    If `reset_mechanism = "zero"`, then :math:`U[t+1]` will be set to `0`
    whenever the neuron emits a spike:

    .. math::

            U[t+1] = (1 - R)(2U[t] - U_{\\rm prev}[t] + \\Delta t^2
            (-\\chi^2 U[t] + I_{\\rm in}[t+1]))

    * :math:`I_{\\rm in}` - Input current
    * :math:`U` - Membrane potential
    * :math:`U_{\\rm prev}` - Previous membrane potential
    * :math:`U_{\\rm thr}` - Membrane threshold
    * :math:`R` - Reset mechanism: if active, :math:`R = 1`, otherwise \
        :math:`R = 0`
    * :math:`\\chi` - Natural frequency (mass term)
    * :math:`\\Delta t` - Time step

    Example::

        import torch
        import torch.nn as nn
        import snntorch as snn

        chi = 0.5
        dt = 0.1

        # Define Network
        class Net(nn.Module):
            def __init__(self):
                super().__init__()

                # initialize layers
                self.fc1 = nn.Linear(num_inputs, num_hidden)
                self.wl1 = snn.WaveLeaky(chi=chi, dt=dt)
                self.fc2 = nn.Linear(num_hidden, num_outputs)
                self.wl2 = snn.WaveLeaky(chi=chi, dt=dt)

            def forward(self, x, mem1, mem_prev1, mem2, mem_prev2):
                cur1 = self.fc1(x)
                spk1, mem1, mem_prev1 = self.wl1(cur1, mem1, mem_prev1)
                cur2 = self.fc2(spk1)
                spk2, mem2, mem_prev2 = self.wl2(cur2, mem2, mem_prev2)
                return mem1, mem_prev1, spk1, mem2, mem_prev2, spk2


    :param chi: natural oscillation frequency (mass term). May be a
        single-valued tensor (equal for all neurons) or multi-valued
        (one per neuron).
    :type chi: float or torch.tensor

    :param dt: integration time step. Smaller values give more accurate
        dynamics but require more steps per oscillation period.
        Should satisfy dt < 2/chi for stability.
    :type dt: float

    :param threshold: Threshold for :math:`mem` to reach in order to
        generate a spike `S=1`. Defaults to 1
    :type threshold: float, optional

    :param spike_grad: Surrogate gradient for the term dS/dU. Defaults to
        None (corresponds to ATan surrogate gradient. See
        `snntorch.surrogate` for more options)
    :type spike_grad: surrogate gradient function from snntorch.surrogate,
        optional

    :param surrogate_disable: Disables surrogate gradients regardless of
        ``spike_grad`` argument. Useful for ONNX compatibility.
        Defaults to False
    :type surrogate_disable: bool, optional

    :param init_hidden: Instantiates state variables as instance variables.
        Defaults to False
    :type init_hidden: bool, optional

    :param inhibition: If `True`, suppresses all spiking other than the
        neuron with the highest state. Defaults to False
    :type inhibition: bool, optional

    :param learn_chi: Option to enable learnable chi. Defaults to False
    :type learn_chi: bool, optional

    :param learn_threshold: Option to enable learnable threshold. Defaults
        to False
    :type learn_threshold: bool, optional

    :param reset_mechanism: Defines the reset mechanism applied to \
    :math:`mem` each time the threshold is met. Reset-by-subtraction: \
        "subtract", reset-to-zero: "zero", none: "none".
        Defaults to "subtract"
    :type reset_mechanism: str, optional

    :param state_quant: If specified, hidden state :math:`mem` is quantized
        to a valid state for the forward pass. Defaults to False
    :type state_quant: quantization function from snntorch.quant, optional

    :param output: If `True` as well as ``init_hidden=True``, states are
        returned when neuron is called. Defaults to False
    :type output: bool, optional

    :param graded_spikes_factor: output spikes are scaled by this value,
        if specified. Defaults to 1.0
    :type graded_spikes_factor: float or torch.tensor

    :param learn_graded_spikes_factor: Option to enable learnable graded
        spikes. Defaults to False
    :type learn_graded_spikes_factor: bool, optional

    :param reset_delay: If `True`, a spike is returned with a one-step
        delay after the threshold is reached. Defaults to True
    :type reset_delay: bool, optional

    Inputs: \\input_, mem_0, mem_prev_0
        - **input_** of shape `(batch, input_size)`: tensor containing input
            features
        - **mem_0** of shape `(batch, input_size)`: tensor containing the
            initial membrane potential for each element in the batch.
        - **mem_prev_0** of shape `(batch, input_size)`: tensor containing
            the previous membrane potential for each element in the batch.

    Outputs: spk, mem_1, mem_prev_1
        - **spk** of shape `(batch, input_size)`: tensor containing the
            output spikes.
        - **mem_1** of shape `(batch, input_size)`: tensor containing the
            next membrane potential for each element in the batch.
        - **mem_prev_1** of shape `(batch, input_size)`: tensor containing
            the current membrane potential (becomes previous on next step).

    Learnable Parameters:
        - **WaveLeaky.chi** (torch.Tensor) - optional learnable frequency
            must be manually passed in, of shape `1` or (input_size).
        - **WaveLeaky.threshold** (torch.Tensor) - optional learnable
            thresholds must be manually passed in, of shape `1` or
            (input_size).
    """

    def __init__(
        self,
        chi,
        dt=0.1,
        threshold=1.0,
        spike_grad=None,
        surrogate_disable=False,
        init_hidden=False,
        inhibition=False,
        learn_chi=False,
        learn_threshold=False,
        reset_mechanism="subtract",
        state_quant=False,
        output=False,
        graded_spikes_factor=1.0,
        learn_graded_spikes_factor=False,
        reset_delay=True,
    ):
        super().__init__(
            threshold=threshold,
            spike_grad=spike_grad,
            surrogate_disable=surrogate_disable,
            init_hidden=init_hidden,
            inhibition=inhibition,
            learn_threshold=learn_threshold,
            reset_mechanism=reset_mechanism,
            state_quant=state_quant,
            output=output,
            graded_spikes_factor=graded_spikes_factor,
            learn_graded_spikes_factor=learn_graded_spikes_factor,
        )

        self._chi_buffer(chi, learn_chi)
        self.dt = dt

        self._init_mem()

        if self.reset_mechanism_val == 0:  # reset by subtraction
            self.state_function = self._base_sub
        elif self.reset_mechanism_val == 1:  # reset to zero
            self.state_function = self._base_zero
        elif self.reset_mechanism_val == 2:  # no reset, pure integration
            self.state_function = self._base_int

        self.reset_delay = reset_delay

    def _chi_buffer(self, chi, learn_chi):
        if not isinstance(chi, torch.Tensor):
            chi = torch.as_tensor([chi], dtype=torch.float)
        if learn_chi:
            self.chi = nn.Parameter(chi)
        else:
            self.register_buffer("chi", chi)

    def _init_mem(self):
        mem = torch.zeros(0)
        mem_prev = torch.zeros(0)
        self.register_buffer("mem", mem, False)
        self.register_buffer("mem_prev", mem_prev, False)

    def reset_mem(self):
        self.mem = torch.zeros_like(self.mem, device=self.mem.device)
        self.mem_prev = torch.zeros_like(
            self.mem_prev, device=self.mem_prev.device
        )
        return self.mem, self.mem_prev

    def forward(self, input_, mem=None, mem_prev=None):

        if mem is not None:
            self.mem = mem
        if mem_prev is not None:
            self.mem_prev = mem_prev

        if self.init_hidden and (mem is not None or mem_prev is not None):
            raise TypeError(
                "`mem` and `mem_prev` should not be passed as arguments "
                "while `init_hidden=True`"
            )

        if not self.mem.shape == input_.shape:
            self.mem = torch.zeros_like(input_, device=self.mem.device)
        if not self.mem_prev.shape == input_.shape:
            self.mem_prev = torch.zeros_like(
                input_, device=self.mem_prev.device
            )

        self.reset = self.mem_reset(self.mem)
        mem_old = self.mem.clone()
        self.mem = self.state_function(input_)

        if self.state_quant:
            self.mem = self.state_quant(self.mem)

        self.mem_prev = mem_old

        if self.inhibition:
            spk = self.fire_inhibition(self.mem.size(0), self.mem)
        else:
            spk = self.fire(self.mem)

        if not self.reset_delay:
            do_reset = spk / self.graded_spikes_factor - self.reset
            if self.reset_mechanism_val == 0:  # reset by subtraction
                self.mem = self.mem - do_reset * self.threshold
            elif self.reset_mechanism_val == 1:  # reset to zero
                self.mem = self.mem - do_reset * self.mem

        if self.output:
            return spk, self.mem, self.mem_prev
        elif self.init_hidden:
            return spk
        else:
            return spk, self.mem, self.mem_prev

    def _base_state_function(self, input_):
        # Klein-Gordon leapfrog:
        # mem_new = 2*mem - mem_prev + dt^2 * (-chi^2 * mem + input)
        dt2 = self.dt * self.dt
        base_fn = (
            2 * self.mem
            - self.mem_prev
            + dt2 * (-self.chi * self.chi * self.mem + input_)
        )
        return base_fn

    def _base_sub(self, input_):
        return self._base_state_function(input_) - self.reset * self.threshold

    def _base_zero(self, input_):
        self.mem = (1 - self.reset) * self.mem
        self.mem_prev = (1 - self.reset) * self.mem_prev
        return self._base_state_function(input_)

    def _base_int(self, input_):
        return self._base_state_function(input_)

    @classmethod
    def detach_hidden(cls):
        """Returns the hidden states, detached from the current graph.
        Intended for use in truncated backpropagation through time where
        hidden state variables are instance variables."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], WaveLeaky):
                cls.instances[layer].mem.detach_()
                cls.instances[layer].mem_prev.detach_()

    @classmethod
    def reset_hidden(cls):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are instance variables.
        Assumes hidden states have a batch dimension already."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], WaveLeaky):
                cls.instances[layer].mem = torch.zeros_like(
                    cls.instances[layer].mem,
                    device=cls.instances[layer].mem.device,
                )
                cls.instances[layer].mem_prev = torch.zeros_like(
                    cls.instances[layer].mem_prev,
                    device=cls.instances[layer].mem_prev.device,
                )
