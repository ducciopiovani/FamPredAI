from typing import List, Set, Dict, Tuple, Optional, Union

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse.linalg.eigen.arpack.arpack import (
    ArpackNoConvergence as _ArpackNoConvergence,
)

import rescomp

# from . import utilities
# from ._version import __version__


class ESNMod(rescomp.ESN):
    """
    Basic ESN (Echo State Network, typical ANN based Reservoir Computer) model based on rescomp.ESN. Slight
    modifications to make more flexible use possible (e.g. ability to have target data not identical to input
    data for t+1). Numerical details are inherited mostly from rescomp.ESN.
    """

    def __init__(self):
        super().__init__()
        self.logger.debug("Create ESNWrapper instance")

    def train_and_predict(
        self,
        x_data: np.array,
        train_sync_steps: int,
        train_steps: int,
        y_data: Union[None, np.array] = None,
        x_external: Union[None, np.array] = None,
        pred_sync_steps: int = 0,
        pred_steps: int = None,
        disc_steps: int = 0,
        **kwargs,
    ) -> Tuple[np.array, np.array]:
        """ Train, then predict the evolution directly following the train data

        Args:
            x_data (np.ndarray): Data used for synchronization, training and
                prediction (start and comparison)
            train_sync_steps (int): Steps to synchronize the reservoir with
                before the 'real' training begins
            train_steps (int): Steps to use for training and fitting w_in
            y_data (np.ndarry): Data used as target in training and as test
                data for prediction. If None
            x_external (np.darray): Data where future values are available and
                which can be used as external input during training as well
                as prediction (e.g. Ramadan).
            pred_sync_steps (int): steps to sync the reservoir with before
                prediction
            pred_steps (int): How many steps to predict the evolution for
            **kwargs: further arguments passed to :func:`~esn.ESN.train` and
                :func:`~esn.ESN.predict`

        Returns:
            tuple: 2-element tuple containing:

            - **y_pred** (*np.ndarray*): Predicted future states
            - **y_test** (*np.ndarray_or_None*): Data taken from the input to
              compare the prediction with. If the prediction were
              "perfect" y_pred and y_test would be equal. Be careful
              though, y_test might be shorter than y_pred, or even None,
              if pred_steps is not None

        """
        x_train, x_pred = rescomp.utilities.train_and_predict_input_setup(
            x_data,
            disc_steps=disc_steps,
            train_sync_steps=train_sync_steps,
            train_steps=train_steps,
            pred_sync_steps=pred_sync_steps,
            pred_steps=pred_steps,
        )

        y_train, y_pred = rescomp.utilities.train_and_predict_input_setup(
            y_data,
            disc_steps=disc_steps,
            train_sync_steps=train_sync_steps,
            train_steps=train_steps,
            pred_sync_steps=pred_sync_steps,
            pred_steps=pred_steps,
        )

        if x_external is not None:
            (
                x_external_train,
                x_external_pred,
            ) = rescomp.utilities.train_and_predict_input_setup(
                x_external,
                disc_steps=disc_steps,
                train_sync_steps=train_sync_steps,
                train_steps=train_steps,
                pred_sync_steps=pred_sync_steps,
                pred_steps=pred_steps,
            )

            x_train = np.append(x_train, x_external_train, 1)

        train_kwargs = rescomp.utilities._remove_invalid_args(self.train, kwargs)
        predict_kwargs = rescomp.utilities._remove_invalid_args(self.predict, kwargs)

        self.train(x_train, y_train, train_sync_steps, **train_kwargs)

        y_pred, y_test = self.predict(
            x_pred,
            x_external=x_external_pred,
            sync_steps=pred_sync_steps,
            pred_steps=pred_steps,
            **predict_kwargs,
        )

        return y_pred, y_test

    def _fit_w_out(
        self, x_train: np.array, r: np.array, y_train: Union[None, np.array] = None
    ) -> np.array:

        """ Fit the output matrix self._w_out after training

        Uses linear regression and Tikhonov regularization.

        Args:
            x_train (np.ndarray): If y_train is None get y_train via x_train[1:]
            r (np.ndarray): reservoir states
            y_train (np.ndarray): Desired prediction from the reservoir states, if None get y_train via x_train[1:]
        Returns:
            np.ndarray: r_gen, generalized nonlinear transformed r

        """
        # Note: in an older version y_train was obtained from x_train in _train_synced, and then
        # parsed directly to _fit_w_out.
        # This is changed in order to allow for an easy override of _fit_w_out in variations of
        # the ESN class (e.g. ESNHybrid), that include other non-rc predictions (e.g. x_new = model(x_old))
        # into an "extended r_gen" = concat(r_gen, model(x)), that is finally fitted to y_train.

        # Note: This is slightly different than the old ESN as y_train was as
        # long as x_train, but shifted by one time step. Hence to get the same
        # results as for the old ESN one has to specify an x_train one time step
        # longer than before. Nonetheless, it's still true that r[t] is
        # calculated from x[t] and used to calculate y[t] (all the same t)

        if y_train is None:
            y_train = x_train[1:]

        self.logger.debug(
            "Fit _w_out according to method %s" % str(self._w_out_fit_flag)
        )

        r_gen = self._r_to_generalized_r(r)

        # If we are using local states we only want to use the core dimensions
        # for the fit of W_out, i.e. the dimensions where the corresponding
        # locality matrix == 2
        if self._loc_nbhd is None:
            self._w_out = np.linalg.solve(
                r_gen.T @ r_gen + self._reg_param * np.eye(r_gen.shape[1]),
                r_gen.T @ y_train,
            ).T
        else:
            self._w_out = np.linalg.solve(
                r_gen.T @ r_gen + self._reg_param * np.eye(r_gen.shape[1]),
                r_gen.T @ y_train[:, self._loc_nbhd == 2],
            ).T

        return r_gen

    def _train_synced(
        self,
        x_train: np.array,
        y_train: Union[None, np.array] = None,
        w_out_fit_flag: str = "simple",
    ) -> Tuple[np.array, np.array]:

        """ Train a synchronized reservoir

        Args:
            x_train (np.ndarray): input to be used for the training, shape (T,d)
            y_train (np.ndarray): Desired prediction from the reservoir states, if None get y_train via x_train[1:] later
            w_out_fit_flag (str): Type of nonlinear transformation applied to
                the reservoir states r to be used during the fit (and future
                prediction

        Returns:
            tuple: 2-element tuple containing:

            - **r** (*np.ndarray*) reservoir states
            - **r_gen** (*np.ndarray*): generalized reservoir states

        """

        self._w_out_fit_flag = self._w_out_fit_flag_synonyms.get_flag(w_out_fit_flag)

        self.logger.debug("Start training")

        # The last value of r is not used for the training,
        # is instead synchronized later during prediction (refactor this carefully!)
        r = self.synchronize(x_train[:-1], save_r=True)

        r_gen = self._fit_w_out(x_train, r, y_train)

        return r, r_gen

    def train(
        self,
        x_train: np.array,
        sync_steps: int,
        y_train: np.array = None,
        reg_param: float = 1e-5,
        w_in_scale: float = 1.0,
        w_in_sparse: bool = True,
        w_in_ordered: bool = False,
        w_in_no_update: bool = False,
        act_fct_flag: str = "tanh_simple",
        bias_scale: float = 0,
        mix_ratio: float = 0.5,
        save_r: bool = False,
        w_out_fit_flag: str = "simple",
        loc_nbhd: Union[np.array, None] = None,
    ) -> None:

        """ Synchronize, then train the reservoir

        Args:
            x_train (np.ndarray): Input data used to synchronize and then train
                the reservoir
            sync_steps (int): How many steps to use for synchronization before
                the prediction starts
            y_train (np.array): Target data, y_train[:sync_steps] is discarded
            reg_param (float): weight for the Tikhonov-regularization term
            w_in_scale (float): maximum absolute value of the (random) w_in
                elements
            w_in_sparse (bool): If true, creates w_in such that one element in
                each row is non-zero (Lu,Hunt, Ott 2018)
            w_in_orderd (bool): If true and w_in_sparse is true, creates w_in
                such that elements are ordered by dimension and number of
                elements for each dimension is equal (as far as possible)
            w_in_no_update (bool): If true and the input matrix W_in does
                already exist from a previous training run, W_in does not get
                updated, regardless of all other parameters concerning it.
            act_fct_flag (int_or_str): Specifies the activation function to be
                used during training (and prediction). Possible flags and their
                synonyms are:

                - "tanh_simple", "simple"
                - "tanh_bias"
                -
            bias_scale (float): Bias to be used in some activation functions
            mix_ratio (float): Ratio of normal tanh vs squared tanh activation
                functions if act_fct_flag "mixed" is chosen
            save_r (bool): If true, saves r(t) internally
            save_input (bool): If true, saves the input data internally
            w_out_fit_flag (str): Type of nonlinear transformation applied to
                the reservoir states r to be used during the fit (and future
                prediction)
            loc_nbhd (np.ndarray): The local neighborhood used for the
                generalized local states approach. For more information, please
                see the docs.
        """
        self._reg_param = reg_param
        self._loc_nbhd = loc_nbhd
        try:
            x_dim = x_train.shape[1]
        except IndexError:
            x_dim = 1
        if self._w_in is not None and w_in_no_update:
            if not self._x_dim == x_dim:
                raise Exception(
                    f"the x_dim specified in create_input_matrix does not match the data x_dim: {self._x_dim} vs {x_dim}"
                )
        else:
            self.create_input_matrix(
                x_dim,
                w_in_scale=w_in_scale,
                w_in_sparse=w_in_sparse,
                w_in_ordered=w_in_ordered,
            )

        self._set_activation_function(act_fct_flag=act_fct_flag, bias_scale=bias_scale)

        if sync_steps != 0:
            x_sync = x_train[:sync_steps]
            x_train = x_train[sync_steps:]
            y_train = y_train[sync_steps:]
            self.synchronize(x_sync)
        else:
            x_sync = None

        self._x_train = x_train

        if save_r:
            self._r_train, self._r_train_gen = self._train_synced(
                x_train, y_train=y_train, w_out_fit_flag=w_out_fit_flag
            )
        else:
            self._train_synced(x_train, y_train=y_train, w_out_fit_flag=w_out_fit_flag)

    def predict(
        self,
        columns_to_predict: Union[np.array, None] = None,
        x_external: Union[np.array, None] = None,
        sync_steps: int = 0,
        pred_steps: Union[int, None] = None,
        save_r: bool = False,
    ) -> Tuple[np.array, np.array]:
        """ Synchronize the reservoir, then predict the system evolution
        Changes self._last_r and self._last_r_gen to stay synchronized to the
        new system state
        Args:
            columns_to_predict (np.ndarray): Input data used to synchronize the reservoir,
                and then used as comparison for the prediction by being returned
                as y_pred in the output. Should also include external data at the end.
            x_external (np.ndarray): Data where future values are available and
                which can be used as external input during training as well
                as prediction (e.g. Ramadan).
            sync_steps (int): How many steps to use for synchronization before
                the prediction starts
            pred_steps (int): How many steps to predict
            save_r (bool): If true, saves r(t) internally
        Returns:
            tuple: 2-element tuple containing:
            - **y_pred** (*np.ndarray*): Predicted future states
        """

        self.logger.debug("Start Prediction")
        self._y_pred = np.zeros((pred_steps, columns_to_predict))

        # taking the last row
        xseed = self._x_train[-1,:]
        # removed x_pred[sync_steps] from the argument
        self._y_pred[0] = self._predict_step(xseed)  # synchronize with last value before prediction

        if save_r:
            self._r_pred = np.zeros((pred_steps, self._network.shape[0]))
            self._r_pred_gen = self._r_to_generalized_r(self._r_pred)
            self._r_pred[0] = self._last_r
            self._r_pred_gen[0] = self._last_r_gen

        for t in range(pred_steps - 1):
            if x_external is None:
                self._y_pred[t + 1] = self._predict_step(self._y_pred[t])
            else:
                self._y_pred[t + 1] = self._predict_step(
                    np.append(self._y_pred[t], x_external[t + 1])
                )
            if save_r:
                self._r_pred[t + 1] = self._last_r
                self._r_pred_gen[t + 1] = self._last_r_gen

        return self._y_pred
    
    def _set_activation_function(self, act_fct_flag, bias_scale=0, mix_ratio=0.5):
        """ Set the activation function corresponding to the act_fct_flag

        Args:
            act_fct_flag (int_or_str): flag corresponding to the activation
                function one wants to use, see :func:`~esn.ESN.train` for a
                list of possible flags
            bias_scale (float): Bias to be used in some activation functions
                (currently only used in :func:`~esn.ESN._act_fct_tanh_bias`)

        """
        self.logger.debug("Set activation function to flag: %s" % act_fct_flag)

        # self._act_fct_flag = act_fct_flag
        self._act_fct_flag = self._act_fct_flag_synonyms.get_flag(act_fct_flag)

        self._bias_scale = bias_scale
        self._bias = self._bias_scale * np.random.uniform(low=-1.0, high=1.0,
                                                          size=self._n_dim)

        if self._act_fct_flag == 0:
            self._act_fct = self._act_fct_tanh_simple
        elif self._act_fct_flag == 1:
            self._act_fct = self._act_fct_tanh_bias
        elif self._act_fct_flag == 2:
            self._act_fct = self._act_fct_tanh_squared
        elif self._act_fct_flag == 3:
            self.setup_mix(mix_ratio)
            self._act_fct = self._act_fct_mixed
        else:
            raise Exception('self._act_fct_flag %s does not have a activation '
                            'function implemented!' % str(self._act_fct_flag))

    def check_if_array(self, x = None):
        if x is not None:
            if not hasattr(x, "__len__"):
                x = np.array([x])
        return x
    def _act_fct_tanh_simple(self, x, r):
        """ Standard activation function of the elementwise np.tanh()

        Args:
            x (np.ndarray): d-dim input
            r (np.ndarray): n-dim network states

        Returns:
            np.ndarray: n-dim

        """

        x = self.check_if_array(x)

        return np.tanh(self._w_in @ x + self._network @ r)

    def _act_fct_tanh_bias(self, x, r):
        """ Activation function of the elementwise np.tanh() with added bias

        Args:
            x (np.ndarray): d-dim input
            r (np.ndarray): n-dim network states

        Returns:
            np.ndarray n-dim

        """
        x = self.check_if_array(x)

        return np.tanh(self._w_in @ x + self._network @ r + self._bias)

    def _act_fct_tanh_squared(self, x, r):
        """ Activation function of the elementwise np.tanh() squared with added
            bias.

        Args:
            x (np.ndarray): d-dim input
            r (np.ndarray): n-dim network states

        Returns:
            np.ndarray n-dim

        """
        x = self.check_if_array(x)

        return np.tanh(self._w_in @ x + self._network @ r + self._bias) ** 2

    def _scale_network(self):
        """ Scale self._network, according to desired spectral radius.

        Can cause problems due to non converging of the eigenvalue evaluation

        Specification done via protected members

        """
        self._network = scipy.sparse.csr_matrix(self._network)
        try:
            eigenvals = scipy.sparse.linalg.eigs(
                self._network, k=1, v0=np.ones(self._n_dim),
                maxiter=10 * self._n_dim)[0]
        except _ArpackNoConvergence:
            self.logger.error('Eigenvalue calculation in scale_network failed!')
            raise

        maximum = np.absolute(eigenvals).max()
        self._network = ((self._n_rad / maximum) * self._network)
    

    
"""

        # Automatically generates a y_test to compare the prediction against, if
        # the input data is longer than the number of synchronization tests
        if sync_steps == 0:
            x_sync = None
            y_test = x_pred[1:]
        elif sync_steps <= x_pred.shape[0]:
            x_sync = x_pred[:sync_steps]
            y_test = x_pred[sync_steps + 1:]
        else:
            x_sync = x_pred[:-1]
            y_test = None

        if save_input:
            self._x_pred_sync = x_sync
            self._y_test = y_test

        if x_sync is not None:
            self.synchronize(x_sync)
"""