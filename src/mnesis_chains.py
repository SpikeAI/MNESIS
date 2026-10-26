"""Core module of the MNESIS library.

Defines the spiking-pattern generators (:class:`SpikingPattern`,
:class:`StochasticSpikingPattern`) and the :class:`HD_SNN` network class --
a recurrent spiking neural network with heterogeneous synaptic delays --
together with its analytical weight initialisation, training
(:meth:`HD_SNN.learn_model`) and inference (:meth:`HD_SNN.forward_pass`)
methods, and a :func:`load` helper for trained checkpoints.
"""

from mnesis_boilerplate import torch, np, nn, OrderedDict, surrogate, snn, snn_utils
from mnesis_boilerplate import get_scores, get_cosine_schedule_with_warmup, SpikeF1scoreLoss
from mnesis_boilerplate import flip_bits, Params


class SpikingPattern:
    """Generate a frozen spiking pattern based on a Bernoulli process.

    The target is a single, reproducible spike raster of shape
    ``(N_pattern, N_neuron, N_time)`` drawn once from a Bernoulli
    distribution of rate ``p_A``, seeded with ``opt.seed`` so that the same
    pattern is obtained across runs.
    """

    def __init__(self):
        self.desc = "Frozen Spike pattern generator"
        self.is_periodic = False

    def init(self, opt, verbose=False):
        """Draw the frozen target pattern.

        Args:
            opt: A :class:`~mnesis_boilerplate.Params` instance providing
                ``N_pattern``, ``N_neuron``, ``N_time``, ``p_A``, ``seed``
                and ``device``.
            verbose: If ``True``, print the shape and mean firing rate of
                the generated target.
        """
        self.opt = opt
        frozen_target_generator = torch.Generator()  # used once to generate the target pattern
        frozen_target_generator.manual_seed(opt.seed)
        p_bias = opt.p_A * torch.ones((opt.N_pattern, opt.N_neuron, opt.N_time))
        self.frozen_target = torch.bernoulli(p_bias, generator=frozen_target_generator)
        self.frozen_target = self.frozen_target.float()
        self.frozen_target = self.frozen_target.to(opt.device)
        if verbose:
            print(f"Target pattern generated with shape {self.frozen_target.shape} and mean {self.frozen_target.mean().item():.3e}")

    def __call__(self, seed=None, verbose=False):
        """Return the frozen target pattern.

        Args:
            seed: Ignored; accepted so that every generator can be called
                with the same signature as :class:`StochasticSpikingPattern`.
            verbose: Ignored, for the same reason.

        Returns:
            torch.Tensor: The stored spike raster, of shape
            ``(N_pattern, N_neuron, N_time)``.
        """
        return self.frozen_target


class StochasticSpikingPattern(SpikingPattern):
    """A stochastic spiking pattern generator.

    Extends :class:`SpikingPattern` with stochastic variability: each call
    returns a new realization of the base pattern in which bits are flipped
    independently with probability ``opt.p_flip`` ("balanced bit flipping"),
    so the marginal firing rate is preserved while the pattern structure is
    stochastically modified.
    """

    def __init__(self): 
        super().__init__()
        self.desc = "Stochastic spike pattern generator"

    def __call__(self, seed=None, verbose=False):
        """Generate a stochastic realization of the spiking pattern.

        Each call returns a new realization of the base pattern where bits
        are independently flipped with probability ``self.opt.p_flip``. The
        balanced-flip operation preserves the marginal frequency while
        introducing temporal and spatial variability in the structure.

        Args:
            seed: Seed for the bit-flip generator; ``None`` draws a fresh
                random seed.
            verbose: If ``True``, report the number of flipped bits.

        Returns:
            torch.Tensor: Stochastic realization of the spiking pattern,
            with the same dimensions as the base pattern.
        """
        return flip_bits(self.frozen_target, p_flip=self.opt.p_flip, seed=seed, verbose=verbose)


class HD_SNN(nn.Module):
    """Recurrent spiking neural network with Heterogeneous Delays (HD-SNN).

    Each of the ``num_delay`` delays of every synapse carries an independent
    learnable weight, gathered in a single tensor ``W`` of shape
    ``(N_neuron, N_neuron * num_delay)``: the membrane of a neuron is
    driven by the last ``num_delay`` spikes of all its presynaptic peers.
    The network is a ``lin`` (the delay-expanded linear layer, no bias)
    followed by ``dropout`` and a leaky integrate-and-fire neuron
    (``snntorch`` ``snn.Leaky``) whose non-differentiable threshold crossing
    is trained through a surrogate gradient.

    Attributes:
        opt: The :class:`~mnesis_boilerplate.Params` configuration.
        pattern_object: The spiking-pattern generator providing the targets.
        net: ``nn.Sequential`` of the ``('lin', 'dropout', 'lif')`` modules.
    """
    def __init__(self, opt, pattern_object):
        """Build the network and initialise the pattern generator.

        Args:
            opt: A :class:`~mnesis_boilerplate.Params` instance.
            pattern_object: A :class:`SpikingPattern`-like object; its
                ``init(opt)`` method is called to produce the targets.
        """
        super().__init__()
        self.opt = opt
        self.pattern_object = pattern_object
        self.pattern_object.init(opt)

        dropout = nn.Dropout(opt.dropout)
        lin = nn.Linear(opt.num_delay*opt.N_neuron, opt.N_neuron, bias=False)
        
        if self.opt.surrogate_name == 'FastSigmoid':
            spike_grad = surrogate.fast_sigmoid(slope=opt.alpha_surrogate)
        elif self.opt.surrogate_name == 'LeakySpikeOperator':
            spike_grad = surrogate.LSO(slope=opt.alpha_surrogate)
        elif self.opt.surrogate_name == 'ATan':
            spike_grad = surrogate.atan(alpha=opt.alpha_surrogate)
        elif self.opt.surrogate_name == 'SpikeRateEscape': # Placeholder from original source typo likely
            spike_grad = surrogate.spike_rate_escape(slope=opt.alpha_surrogate)
        elif self.opt.surrogate_name == 'Sigmoid':
            spike_grad = surrogate.sigmoid(slope=opt.alpha_surrogate)
        else:
            spike_grad = surrogate.fast_sigmoid(slope=opt.alpha_surrogate)

        lif = snn.Leaky(beta=torch.tensor(opt.lif_beta, dtype=torch.float32), 
                        threshold=torch.tensor(opt.lif_threshold, dtype=torch.float32),
                        learn_beta=opt.learn_beta, learn_threshold=opt.learn_threshold, output=False,
                        reset_mechanism=opt.reset_mechanism, spike_grad=spike_grad)

        self.net = nn.Sequential(OrderedDict([('lin', lin), ('dropout', dropout), ('lif', lif)]))
        self.net = self.net.to(opt.device)
          
    def forward_pass(self, input_spikes, reset_spikes=None):
        """Unroll the network over time and record its activations.

        Starting at ``t = num_delay``, each neuron receives the last
        ``num_delay`` spikes of every neuron, taken from the recurrent
        output, the external input and (optionally) a reset mask, combined
        as ``(A + B - C).clamp(0, 1)`` and ravelled into the delay-expanded
        linear layer before the leaky integrate-and-fire step.

        On MPS the loop synchronises every ``sync_every`` steps: without it,
        a long run (about 1e5 steps) piles up all its kernels in a single
        Metal command buffer, which the driver rejects as a whole
        (``kIOGPUCommandBufferCallbackErrorOutOfMemory``) while PyTorch
        silently returns all-zero tensors.

        Args:
            input_spikes: External spikes of shape
                ``(N_pattern, N_neuron, N_time)``.
            reset_spikes: Spikes to clamp off from the recurrent drive,
                same shape as ``input_spikes``; ``None`` means no reset.

        Returns:
            tuple: ``(current, mem_rec, spikes)`` -- the post-synaptic
            current, the membrane potential and the output spikes, each of
            shape ``(N_pattern, N_neuron, N_time)``.
        """
        input_spikes = input_spikes.to(self.opt.device).detach()
        if reset_spikes is None: reset_spikes = torch.zeros_like(input_spikes)

        with torch.no_grad():
            snn_utils.reset(self.net)

        device, dtype = self.opt.device, torch.float32
        N_pattern = input_spikes.shape[0]
        N_time = input_spikes.shape[-1]
        current = torch.zeros(N_pattern, self.opt.N_neuron, N_time, device=device, dtype=dtype)
        spikes  = torch.zeros(N_pattern, self.opt.N_neuron, N_time, device=device, dtype=dtype)
        mem_rec = torch.zeros(N_pattern, self.opt.N_neuron, N_time, device=device, dtype=dtype)
        mem = self.net.lif.init_leaky()
        sync_every = 1000 if device.type == 'mps' else 0 # number of steps between two MPS synchronisations (0 = never)

        for t in range(self.opt.num_delay, N_time):
            spike_window_A = spikes[:, :, (t - self.opt.num_delay):t]
            spike_window_B = input_spikes[:, :, (t - self.opt.num_delay):t]
            spike_window_C = reset_spikes[:, :, (t - self.opt.num_delay):t]
            spike_window = (spike_window_A + spike_window_B - spike_window_C).clamp(0, 1)
            raveled_spks = spike_window.reshape(N_pattern, self.opt.N_neuron * self.opt.num_delay)
            cur = self.net.lin(raveled_spks)
            cur = self.net.dropout(cur)
            spk, mem = self.net.lif(cur, mem)
            current[:, :, t] = cur
            spikes[:, :, t] = spk
            mem_rec[:, :, t] = mem
            # flush the queued kernels before the command buffer grows too large to be accepted
            if sync_every and (t % sync_every == 0): torch.mps.synchronize()

        return current, mem_rec, spikes

    def get_W_init(self):
        """Compute the closed-form (analytical) weight initialisation.

        The target patterns are cast as a linear regression problem: every
        sliding window of ``num_delay`` past spikes (the "context") should
        predict the next time step of activity. Two orthogonal flags select
        the estimator:

        - ``do_deconv``: deconvolve the LIF membrane, i.e. regress onto the
          input current ``theta_0 * (s*(t) - beta * s*(t-1))`` rather than
          onto the raw target spikes.
        - ``do_pinv``: solve exactly with the (CPU-computed) pseudo-inverse
          ``W = pinv(C) T``; otherwise use the normalised Hebbian
          cross-correlation ``T^T C / <||c||^2>``.

        The regression is fitted on the frozen pattern, the noise-free
        reference that every evaluation scores against, so that the
        initialisation is deterministic for a given ``opt.seed``.

        Returns:
            torch.Tensor: The initial weights for ``net.lin``, of shape
            ``(N_neuron, N_neuron * num_delay)``.
        """
        target = self.pattern_object.frozen_target
        windows = target[:, :, :-1].unfold(dimension=2, size=self.opt.num_delay, step=1)
        windows  = windows.permute(0, 2, 1, 3).contiguous()
        batch    = self.opt.N_pattern * (self.opt.N_time - self.opt.num_delay)
        contexts = windows.reshape(batch, self.opt.N_neuron * self.opt.num_delay)
            
        if self.opt.do_deconv:
            deconvolved = target - self.opt.lif_beta * torch.roll(target, 1, dims=-1)
            raw_targets = deconvolved[:, :, self.opt.num_delay:]
        else:
            raw_targets = target[:, :, self.opt.num_delay:]

        targets = raw_targets.permute(0, 2, 1).reshape(batch, self.opt.N_neuron)

        if self.opt.do_pinv:
            contexts_cpu, targets_cpu = contexts.cpu(), targets.cpu()
            X_pinv = torch.linalg.pinv(contexts_cpu)
            W_init = torch.matmul(X_pinv, targets_cpu)
            W_init = W_init.transpose(0, 1).to(contexts.device)
        else:
            norm = (contexts * contexts).sum(dim=0).mean().clamp(min=1e-8)
            W_init = torch.matmul(targets.t(), contexts) / norm

        return W_init
        
    def update_weight(self):
        """Set ``net.lin.weight`` to the analytical initialisation.

        Copies :meth:`get_W_init` into the linear layer in-place, under
        ``torch.no_grad()``.
        """
        with torch.no_grad():            
            W_init = self.get_W_init()
            self.net.lin.weight.copy_(W_init)
            
    def get_input_spikes(self, target, p_A=None, N_pretime=None, N_trigger_time=None, N_time=None, seed=None):
        """Build the trigger input spikes for the network.

        The input concatenates a ``N_pretime`` chunk of spontaneous
        Bernoulli activity at rate ``p_A``, then the trigger window of
        ``num_delay`` steps, which shows the last ``N_trigger_time`` steps
        of its spiking-motif window (target steps ``num_delay -
        N_trigger_time`` to ``num_delay - 1``); its first ``num_delay -
        N_trigger_time`` steps prolong the spontaneous activity. Any
        remaining time bins stay silent. The total length is
        ``N_time + 2*N_pretime``.

        A partial trigger is aligned on the end of the window because the
        readout is only trained on complete spiking-motif windows, which
        end where the replay starts: the partial window shown to the first
        evoked step is then a subset of a trained context, not a context
        shifted into the spontaneous activity.

        Args:
            target: The memorised pattern of shape
                ``(N_pattern, N_neuron, N_time)``.
            p_A: Spontaneous firing rate; defaults to ``opt.p_A``.
            N_pretime: Length of the spontaneous prefix; defaults to
                ``opt.N_pretime``.
            N_trigger_time: Number of steps of the cue, taken at the end
                of the trigger window (time bins, ``0 <= N_trigger_time <=
                num_delay``); defaults to ``opt.num_delay`` (full trigger).
            N_time: Length of the memory stretch; defaults to ``opt.N_time``.
            seed: Seed for the spontaneous activity; ``None`` draws from the
                global ``torch`` generator (seeded by ``Params``).

        Returns:
            torch.Tensor: Detached input spikes of shape
            ``(N_pattern, N_neuron, N_time + 2*N_pretime)``.
        """
        if p_A is None: p_A = self.opt.p_A
        if N_pretime is None: N_pretime = self.opt.N_pretime
        if N_trigger_time is None: N_trigger_time = self.opt.num_delay
        if N_time is None: N_time = self.opt.N_time
        assert 0 <= N_trigger_time <= self.opt.num_delay, f"N_trigger_time={N_trigger_time} must lie in [0, num_delay={self.opt.num_delay}]"
        generator = None if seed is None else torch.Generator().manual_seed(int(seed))
        # first step of the cue: the spontaneous activity runs up to it
        N_start = N_pretime + self.opt.num_delay - N_trigger_time

        input_spikes = torch.zeros((self.opt.N_pattern, self.opt.N_neuron, N_time+2*N_pretime))
        # spontaneous activity before the cue (same draw as before for a full trigger)
        input_spikes[:, :, :N_start] = torch.bernoulli(p_A * torch.ones((self.opt.N_pattern, self.opt.N_neuron, N_start)), generator=generator)
        # the cue: the last N_trigger_time steps of the spiking-motif window
        input_spikes[:, :, N_start:(N_pretime+self.opt.num_delay)] = target[:, :, (self.opt.num_delay-N_trigger_time):self.opt.num_delay]
        return input_spikes.to(self.opt.device).detach()

    def learn_model(self, verbose=True):
        """Train the delay weights by surrogate-gradient BPTT.

        At each epoch a fresh noisy realisation is drawn from
        ``pattern_object`` (data augmentation), the network is unrolled
        through :meth:`forward_pass` on the cued input from
        :meth:`get_input_spikes`, and the loss (``SpikeF1scoreLoss`` by
        default, or ``MSELoss``) between the output and that realisation is
        evaluated only after the spontaneous pre-time and the trigger window.
        The learning rate follows a cosine decay with linear warmup. An
        evaluation pass (dropout disabled), cued by another realisation and
        scored against the frozen pattern, logs loss, precision, recall and
        F1 every ``num_epochs // 64`` epochs. All realisations are seeded
        from ``opt.seed``, so a run is reproducible.

        Args:
            verbose: If ``True``, print periodic training logs.
        """
        if self.opt.loss_name == 'SpikeF1scoreLoss':
            loss_fn = SpikeF1scoreLoss()
        elif self.opt.loss_name == 'MSELoss':
            loss_fn = nn.MSELoss()

        self.net = self.net.to(self.opt.device)
        optimizer_dict = dict(lr=self.opt.base_lr)
        if self.opt.optimizer=='adam': 
            optimizer = torch.optim.Adam(self.net.parameters(), betas=(1-self.opt.delta1, 1-self.opt.delta2), **optimizer_dict)
        elif self.opt.optimizer=='adamw': 
            optimizer = torch.optim.AdamW(self.net.parameters(), betas=(1-self.opt.delta1, 1-self.opt.delta2), **optimizer_dict)
        elif self.opt.optimizer=='sparseadam': 
            optimizer = torch.optim.AdamW(self.net.parameters(), betas=(1-self.opt.delta1, 1-self.opt.delta2), **optimizer_dict)
        elif self.opt.optimizer=='sgd': 
            optimizer = torch.optim.SGD(self.net.parameters(),  momentum=1-self.opt.delta1, dampening=1-self.opt.delta2, **optimizer_dict)
        elif self.opt.optimizer=='rmsprop': 
            optimizer = torch.optim.RMSprop(self.net.parameters(), momentum=1-self.opt.delta1, alpha=1-self.opt.delta2, **optimizer_dict)
        elif self.opt.optimizer=='adadelta': 
            optimizer = torch.optim.Adadelta(self.net.parameters(), rho=1-self.opt.delta1, **optimizer_dict)
        else:
            raise(ValueError(f'Unknown optimizer {self.opt.optimizer}'))

        scheduler = get_cosine_schedule_with_warmup(optimizer, self.opt.num_warmup_epochs, self.opt.num_epochs, self.opt.final_lr/self.opt.base_lr)

        loss_val, precision, recall, f1_score = [], [], [], []
        log_interval = max(self.opt.num_epochs // 64, 1)
        # one (train, validation) seed pair per epoch, drawn from opt.seed: large random
        # integers, so they do not coincide with the evaluation seeds opt.seed + i of the notebooks
        seeds = np.random.default_rng(self.opt.seed).integers(2**31, size=(self.opt.num_epochs, 2))
        target_frozen = self.pattern_object.frozen_target[:, :, self.opt.num_delay:]

        for i_step in range(self.opt.num_epochs):
            self.net.train()
            seed_train, seed_val = seeds[i_step].tolist()
            # the pattern that we wish to memorize
            target = self.pattern_object(seed=seed_train) # NOTE: we assume that the pattern generator can generate a new pattern each time it is called
            # the input spikes that we feed to the network, which includes pre-time spontaneous activity (padding) and the target pattern just for the trigger window
            input_spikes = self.get_input_spikes(target=target, seed=seed_train).detach()
            optimizer.zero_grad()
            # the optimal output spikes that the network produces in response to the input spikes
            _, _, output_spikes = self.forward_pass(input_spikes)
            # the loss is computed only on the output spikes that correspond to the target pattern, i.e. after the pre-time spontaneous activity and after the trigger window
            loss_train = loss_fn(output_spikes[:, :, (self.opt.N_pretime+self.opt.num_delay):(self.opt.N_time+self.opt.N_pretime)], 
                                 target[:, :, self.opt.num_delay:])
            loss_train.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                self.net.eval()
                # a fresh noisy realisation cues the network...
                target = self.pattern_object(seed=seed_val)
                # the input spikes
                input_spikes = self.get_input_spikes(target=target, seed=seed_val).detach()
                # the optimal output spikes that the network produces in response to the input spikes
                _, _, output_spikes = self.forward_pass(input_spikes)
                output_spikes_trimmed = output_spikes[:, :, (self.opt.N_pretime+self.opt.num_delay):(self.opt.N_time+self.opt.N_pretime)]
                # ...and the replay is scored against the noise-free frozen pattern, as in the notebooks
                loss_val_ = loss_fn(output_spikes_trimmed, target_frozen)
                loss_val.append(loss_val_.item())
                precision_, recall_, f1_score_ = get_scores(output_spikes_trimmed, target_frozen)
                precision.append(precision_.cpu()) 
                recall.append(recall_.cpu())
                f1_score.append(f1_score_.cpu())

            if verbose and ((i_step + 1) % log_interval == 0):
                print(f'Train Epoch [{i_step+1:06d}/{self.opt.num_epochs:06d}]\t| Loss = {np.mean(loss_val):.3e}\t| precision = {np.mean(precision):.3f}\t| recall = {np.mean(recall):.3f}\t| f1_score = {np.mean(f1_score):.3f}\t| ')
                loss_val, precision, recall, f1_score = [], [], [], []

def load(opt, model_filename, pattern_object=None):
    """Load a trained checkpoint into a fresh :class:`HD_SNN`.

    Rebuild the network (re-initialising the pattern generator), restore
    the saved ``net`` state dict and switch it to evaluation mode.

    Args:
        opt: A :class:`~mnesis_boilerplate.Params` instance.
        model_filename: Path to a ``.pth`` state dict saved from ``hd.net``.
        pattern_object: Generator providing the targets; defaults to a
            :class:`StochasticSpikingPattern`.

    Returns:
        HD_SNN: The trained network in ``eval()`` mode.
    """
    if pattern_object is None:
        pattern_object = StochasticSpikingPattern()

    hd = HD_SNN(opt, pattern_object=pattern_object)
    hd.net.to(hd.opt.device)
    model_state_dict = torch.load(model_filename, map_location=torch.device(hd.opt.device))
    hd.net.load_state_dict(model_state_dict)
    hd.net.eval()
    return hd
