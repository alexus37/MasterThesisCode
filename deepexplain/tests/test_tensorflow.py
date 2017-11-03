from unittest import TestCase
import pkg_resources
import logging, os
import tensorflow as tf
import numpy as np


from deepexplain.tensorflow import DeepExplain


def simple_model(activation, session):
    X = tf.placeholder("float", [None, 2])
    w1 = tf.Variable(initial_value=[[1.0, -1.0], [-1.0, 1.0]])
    b1 = tf.Variable(initial_value=[1.5, -1.0])
    w2 = tf.Variable(initial_value=[[1.1, 1.4], [-0.5, 1.0]])
    b2 = tf.Variable(initial_value=[0.0, 2.0])

    layer1 = activation(tf.matmul(X, w1) + b1)
    out = tf.matmul(layer1, w2) + b2
    session.run(tf.global_variables_initializer())
    return X, out


def simpler_model(session):
    """
    Implements ReLU( ReLU(x1 - 1) - ReLU(x2) )
    :
    """
    X = tf.placeholder("float", [None, 2])
    w1 = tf.Variable(initial_value=[[1.0, 0.0], [0.0, 1.0]], trainable=False)
    b1 = tf.Variable(initial_value=[-1.0, 0], trainable=False)
    w2 = tf.Variable(initial_value=[[1.0], [-1.0]], trainable=False)
    logging.critical (w2)

    l1 = tf.nn.relu(tf.matmul(X, w1) + b1)
    out = tf.nn.relu(tf.matmul(l1, w2))
    session.run(tf.global_variables_initializer())
    return X, out


def train_xor(session):
    # Since setting seed is not always working on TF, initial weights values are hardcoded for reproducibility
    X = tf.placeholder("float", [None, 2])
    Y = tf.placeholder("float", [None, 1])
    w1 = tf.Variable(initial_value=[[0.10711301, -0.0987727], [-1.57625198, 1.34942603]])
    b1 = tf.Variable(initial_value=[-0.30955192, -0.14483099])
    w2 = tf.Variable(initial_value=[[0.69259691], [-0.16255915]])
    b2 = tf.Variable(initial_value=[1.53952825])

    l1 = tf.nn.relu(tf.matmul(X, w1) + b1)
    out = tf.matmul(l1, w2) + b2
    session.run(tf.global_variables_initializer())

    # Define loss and optimizer
    loss = tf.reduce_mean(tf.losses.mean_squared_error(Y, out))
    train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    # Generate dataset random
    np.random.seed(10)
    x = np.random.randint(0, 2, size=(10, 2))
    y = np.expand_dims(np.logical_or(x[:, 0], x[:, 1]), -1)
    l = None
    for _ in range(100):
        l, _, = session.run([loss, train_step], feed_dict={X: x, Y: y})
        #logging.critical(l)
    #logging.critical('Done')
    return np.abs(l - 0.1) < 0.01


class TestDeepExplainGeneralTF(TestCase):

    def setUp(self):
        self.session = tf.Session()

    def tearDown(self):
        self.session.close()
        tf.reset_default_graph()

    def test_tf_available(self):
        try:
            pkg_resources.require('tensorflow>=1.0')
        except Exception:
            self.fail("Tensorflow requirement not met")

    def test_simple_model(self):
        X, out = simple_model(tf.nn.relu, self.session)
        xi = np.array([[1, 0]])
        r = self.session.run(out, {X: xi})
        self.assertEqual(r.shape, xi.shape)
        np.testing.assert_equal(r[0], [2.75,  5.5])

    def test_simpler_model(self):
        X, out = simpler_model(self.session)
        xi = np.array([[3.0, 1.0]])
        r = self.session.run(out, {X: xi})
        self.assertEqual(r.shape, (xi.shape[0], 1))
        np.testing.assert_equal(r[0], [1.0])

    def test_training(self):
        session = tf.Session()
        r = train_xor(session)
        self.assertTrue(r)

    def test_context(self):
        """
        DeepExplain overrides nonlinearity gradient
        """
        # No override
        from deepexplain.tensorflow import DeepExplain

        X = tf.placeholder("float", [None, 1])
        activations = {'Relu': tf.nn.relu,
                       'Sigmoid': tf.nn.sigmoid,
                       'Softplus': tf.nn.softplus,
                       'Tanh': tf.nn.tanh}
        for name, f in activations.items():
            x1 = f(X)
            x1_g = tf.gradients(x1, X)[0]
            self.assertEqual(x1_g.op.type, '%sGrad' % name)

        # Override (note: that need to pass graph! Multiple thread testing??)
        with DeepExplain(graph=tf.get_default_graph(), sess=self.session) as de:
            for name, f in activations.items():
                # Gradients of nonlinear ops are overriden
                x2 = f(X)
                self.assertEqual(x2.op.get_attr('_gradient_op_type').decode('utf-8'), 'DeepExplainGrad')

    def test_override_as_default(self):
        """
        In DeepExplain context, nonlinearities behave as default, including training time
        """
        with DeepExplain(graph=tf.get_default_graph(), sess=self.session) as de:
            r = train_xor(self.session)
            self.assertTrue(r)


class TestDummyMethod(TestCase):

    def setUp(self):
        self.session = tf.Session()

    def tearDown(self):
        self.session.close()
        tf.reset_default_graph()

    def test_dummy_zero(self):
        with DeepExplain(graph=tf.get_default_graph(), sess=self.session) as de:
            X, out = simple_model(tf.nn.sigmoid, self.session)
            xi = np.array([[10, -10]])
            attributions = de.explain('zero', out, X, xi)
            self.assertEqual(attributions.shape, xi.shape)
            np.testing.assert_almost_equal(attributions[0], [0.0, 0.0], 10)

    def test_gradient_restored(self):
        with DeepExplain(graph=tf.get_default_graph(), sess=self.session) as de:
            X, out = simple_model(tf.nn.sigmoid, self.session)
            xi = np.array([[10, -10]])
            de.explain('zero', out, X, xi)
            r = train_xor(self.session)
            self.assertTrue(r)


class TestSaliencyMethod(TestCase):

    def setUp(self):
        self.session = tf.Session()

    def tearDown(self):
        self.session.close()
        tf.reset_default_graph()

    def test_saliency_method(self):
        with DeepExplain(graph=tf.get_default_graph(), sess=self.session) as de:
            X, out = simpler_model( self.session)
            xi = np.array([[-10, -5], [3, 1]])
            attributions = de.explain('saliency', out, X, xi)
            self.assertEqual(attributions.shape, xi.shape)
            np.testing.assert_almost_equal(attributions, [[0.0, 0.0], [1.0, 1.0]], 10)

class TestGradInputMethod(TestCase):

    def setUp(self):
        self.session = tf.Session()

    def tearDown(self):
        self.session.close()
        tf.reset_default_graph()

    def test_saliency_method(self):
        with DeepExplain(graph=tf.get_default_graph(), sess=self.session) as de:
            X, out = simpler_model( self.session)
            xi = np.array([[-10, -5], [3, 1]])
            attributions = de.explain('grad*input', out, X, xi)
            self.assertEqual(attributions.shape, xi.shape)
            np.testing.assert_almost_equal(attributions, [[0.0, 0.0], [3.0, -1.0]], 10)

class TestIntegratedGradientsMethod(TestCase):

    def setUp(self):
        self.session = tf.Session()

    def tearDown(self):
        self.session.close()
        tf.reset_default_graph()

    def test_int_grad(self):
        with DeepExplain(graph=tf.get_default_graph(), sess=self.session) as de:
            X, out = simpler_model( self.session)
            xi = np.array([[-10, -5], [3, 1]])
            attributions = de.explain('intgrad', out, X, xi)
            self.assertEqual(attributions.shape, xi.shape)
            np.testing.assert_almost_equal(attributions, [[0.0, 0.0], [1.5, -0.5]], 1)

    def test_int_grad_higher_precision(self):
        with DeepExplain(graph=tf.get_default_graph(), sess=self.session) as de:
            X, out = simpler_model( self.session)
            xi = np.array([[-10, -5], [3, 1]])
            attributions = de.explain('intgrad', out, X, xi, steps=500)
            self.assertEqual(attributions.shape, xi.shape)
            np.testing.assert_almost_equal(attributions, [[0.0, 0.0], [1.5, -0.5]], 2)

    def test_int_grad_baseline(self):
        with DeepExplain(graph=tf.get_default_graph(), sess=self.session) as de:
            X, out = simpler_model(self.session)
            xi = np.array([[-10, -5], [3, 1]])
            attributions = de.explain('intgrad', out, X, xi, baseline=xi)
            self.assertEqual(attributions.shape, xi.shape)
            np.testing.assert_almost_equal(attributions, [[0.0, 0.0], [0.0, 0.0]], 5)