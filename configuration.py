from typing import Collection, Mapping
from dataclasses import FrozenInstanceError, asdict, dataclass, field, replace

class AdapterConfigBase(Mapping):
    """
    Base class for all adaptation methods. This class does not define specific configuration keys, but only provides
    some common helper methods.
    Args:
        architecture (str, optional): The type of adaptation method defined by the configuration.
    """

    architecture= None

    def __init__(self):
        raise TypeError("AdapterConfigBase is an abstract class and cannot be instantiated.")

    # We want to emulate a simple form of immutability while keeping the ability to add custom attributes.
    # Therefore, we don't allow changing attribute values if set once.
    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise FrozenInstanceError()
        else:
            object.__setattr__(self, name, value)

    def __delattr__(self, name):
        raise FrozenInstanceError()

    def __getitem__(self, key):
        return self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __eq__(self, other):
        return self.to_dict() == other.to_dict()

    def to_dict(self):
        """Converts the config class to a Python dict."""
        return asdict(self)

    def replace(self, **changes):
        """Returns a new instance of the config class with the specified changes applied."""
        return replace(self, **changes)

    @classmethod
    def from_dict(cls, config):
        """Creates a config class from a Python dict."""
        if isinstance(config, AdapterConfigBase):
            return config

        # the constructor does not accept additional kwargs, so add them separately
        defined_kwargs, new_kwargs = {}, {}
        for k, v in config.items():
            if k in cls.__dataclass_fields__.keys():
                defined_kwargs[k] = v
            else:
                new_kwargs[k] = v
        obj = cls(**defined_kwargs)
        for k, v in new_kwargs.items():
            setattr(obj, k, v)
        return obj

    @staticmethod
    def _get_config_class(config_dict):
        """
        Returns the matching config class for the given config dict based on its "architecture" key.
        """
        architecture = config_dict.get("architecture", None)
        # if architecture == "prefix_tuning":
        #     cls_new = PrefixTuningConfig
        # elif architecture == "lora":
        #     cls_new = LoRAConfig
        # elif architecture == "union":
        #     cls_new = ConfigUnion
        # else:
        cls_new = AdapterConfig

        return cls_new

    # @classmethod
    # def load(cls, config: Union[dict, str], download_kwargs=None, **kwargs):
    #     """
    #     Loads a given adapter configuration specifier into a full AdapterConfigBase instance.
    #     Args:
    #         config (Union[dict, str]): The configuration to load. Can be either:
    #             - a dictionary representing the full config
    #             - an identifier string available in ADAPTER_CONFIG_MAP
    #             - the path to a file containing a full adapter configuration
    #             - an identifier string available in Adapter-Hub
    #     Returns:
    #         dict: The resolved adapter configuration dictionary.
    #     """
    #     if not config:
    #         return None
    #     # if force_download is set, skip the local map
    #     if download_kwargs and download_kwargs.get("force_download", False):
    #         local_map = None
    #     else:
    #         local_map = ADAPTER_CONFIG_MAP
    #     if download_kwargs:
    #         config_dict = resolve_adapter_config(config, local_map=local_map, **download_kwargs)
    #     else:
    #         config_dict = resolve_adapter_config(config, local_map=local_map)
    #     # convert back to dict to allow attr overrides
    #     if isinstance(config_dict, AdapterConfigBase):
    #         cls_new = config_dict.__class__
    #         config_dict = config_dict.to_dict()
    #     else:
    #         cls_new = cls._get_config_class(config_dict)
    #     # The check for "None" is necessary because of the example script flags.
    #     config_dict.update((k, v) for k, v in kwargs.items() if v is not None)
    #     return cls_new.from_dict(config_dict)


#@dataclass(eq=False)
class AdapterConfig(AdapterConfigBase):
    """
    Base class that models the architecture of an adapter.
    Args:
        mh_adapter (:obj:`bool`): If True, add adapter modules after the multi-head attention block of each layer.
        output_adapter (:obj:`bool`): If True, add adapter modules after the output FFN of each layer.
        reduction_factor (:obj:`float` or :obj:`Mapping`):
            Either a scalar float (> 0) specifying the reduction factor for all layers or a mapping specifying the
            reduction_factor for individual layers. If not all layers are represented in the mapping a default value
            should be given e.g. {'1': 8, '6': 32, 'default': 16}. Specifying a reduction factor < 1 will result in an
            up-projection layer.
        non_linearity (:obj:`str`): The activation function to use in the adapter bottleneck.
        original_ln_before (:obj:`bool`, optional):
            If True, apply layer pre-trained normalization and residual connection before the adapter modules. Defaults
            to False. Only applicable if :obj:`is_parallel` is False.
        original_ln_after (:obj:`bool`, optional):
            If True, apply pre-trained layer normalization and residual connection after the adapter modules. Defaults
            to True.
        ln_before (:obj:`bool`, optional): If True, add a new layer normalization before the adapter bottleneck.
            Defaults to False.
        ln_after (:obj:`bool`, optional): If True, add a new layer normalization after the adapter bottleneck.
            Defaults to False.
        init_weights (:obj:`str`, optional): Initialization method for the weights of the adapter modules.
            Currently, this can be either "bert" (default) or "mam_adapter".
        is_parallel (:obj:`bool`, optional): If True, apply adapter transformations in parallel.
            By default (False), sequential application is used.
        scaling (:obj:`float` or :obj:`str`, optional):
            Scaling factor to use for scaled addition of adapter outputs as done by He et al. (2021). Can bei either a
            constant factor (float) or the string "learned", in which case the scaling factor is learned. Defaults to
            1.0.
        use_gating (:obj:`bool`, optional):
            Place a trainable gating module besides the added parameter module to control module activation. This is
            e.g. used for UniPELT. Defaults to False.
        residual_before_ln (:obj:`bool`, optional):
            If True, take the residual connection around the adapter bottleneck before the layer normalization. Only
            applicable if :obj:`original_ln_before` is True.
        adapter_residual_before_ln (:obj:`bool`, optional):
            If True, apply the residual connection around the adapter modules before the new layer normalization within
            the adapter. Only applicable if :obj:`ln_after` is True and :obj:`is_parallel` is False.
        inv_adapter (:obj:`str`, optional):
            If not None (default), add invertible adapter modules after the model embedding layer. Currently, this can
            be either "nice" or "glow".
        inv_adapter_reduction_factor (:obj:`float`, optional):
            The reduction to use within the invertible adapter modules. Only applicable if :obj:`inv_adapter` is not
            None.
        cross_adapter (:obj:`bool`, optional):
            If True, add adapter modules after the cross attention block of each decoder layer in an encoder-decoder
            model. Defaults to False.
        leave_out (:obj:`List[int]`, optional):
            The IDs of the layers (starting at 0) where NO adapter modules should be added.
        phm_layer (:obj:`bool`, optional): If True the down and up projection layers are a PHMLayer.
            Defaults to False
        phm_dim (:obj:`int`, optional): The dimension of the phm matrix.
            Defaults to None.
        shared_phm_rule (:obj:`bool`, optional): Whether the phm matrix is shared across all layers.
            Defaults to True
        factorized_phm_rule (:obj:`bool`, optional):
            Whether the phm matrix is factorized into a left and right matrix. Defaults to False.
        learn_phm (:obj:`bool`, optional): Whether the phm matrix should be learned during training.
            Defaults to True
        factorized_phm_W (:
            obj:`bool`, optional): Whether the weights matrix is factorized into a left and right matrix. Defaults to
            True
        shared_W_phm (:obj:`bool`, optional): Whether the weights matrix is shared across all layers.
            Defaults to False.
        phm_c_init (:obj:`str`, optional): The initialization function for the weights of the phm matrix.
            The possible values are `["normal", "uniform"]`. Defaults to `normal`.
        phm_init_range (:obj:`float`, optional): std for initializing phm weights if `phm_c_init="normal"`.
            Defaults to 0.0001.
        hypercomplex_nonlinearity (:obj:`str`, optional):
            This specifies the distribution to draw the weights in the phm layer from. Defaults to `glorot-uniform`.
        phm_rank (:obj:`int`, optional):
            If the weight matrix is factorized this specifies the rank of the matrix. E.g. the left matrix of the down
            projection has the shape (phm_dim, _in_feats_per_axis, phm_rank) and the right matrix (phm_dim, phm_rank,
            _out_feats_per_axis). Defaults to 1
        phm_bias (:obj:`bool`, optional):
            If True the down and up projection PHMLayer has a bias term. If `phm_layer` is False this is ignored.
            Defaults to True
    """

    # Required options
    mh_adapter= True
    output_adapter= True

    reduction_factor= 16
    non_linearity = 'relu'

    # Options with defaults
    original_ln_before = False
    original_ln_after= True
    ln_before = False
    ln_after= False
    init_weights= "bert"
    is_parallel = False
    scaling =1.0
    use_gating= False
    residual_before_ln= True
    adapter_residual_before_ln= False
    inv_adapter= None
    inv_adapter_reduction_factor= None
    cross_adapter= False
    leave_out = field(default_factory=list)
    phm_layer= False
    phm_dim: int = 4
    factorized_phm_W = True
    shared_W_phm = False
    shared_phm_rule= True
    factorized_phm_rule = False
    phm_c_init= "normal"
    phm_init_range = 0.0001
    learn_phm = True
    hypercomplex_nonlinearity= "glorot-uniform"
    phm_rank = 1
    phm_bias = True

    # We want to emulate a simple form of immutability while keeping the ability to add custom attributes.
    # Therefore, we don't allow changing attribute values if set once.
    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise FrozenInstanceError()
        elif name == "invertible_adapter":
            # This is for backwards compatibility. In v1, invertible adapters were specified in a nested config dict.
            # Now, we have two config keys directly in the adapter config.
            if value:
                object.__setattr__(self, "inv_adapter", value["block_type"])
                object.__setattr__(self, "inv_adapter_reduction_factor", value["reduction_factor"])
        else:
            object.__setattr__(self, name, value)

