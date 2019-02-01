import datetime
import gc
import os
import sys

# add current module to the sys path, otherwise terminal execution won't recognize this module
sys.path.insert(0, os.path.curdir)

import re
import time
from random import random
import numpy as np
import tensorflow as tf

from app.nlpapp import NLPApp
from input_helpers import InputHelper
from siamese_network import SiameseLSTM
from siamese_network_semantic import SiameseLSTMw2v


# from app import NLPApp
class MetricOps:
	def __init__(self, tr_op_set, train_summary_op, dev_summary_op, train_summary_writer, dev_summary_writer):
		self.tr_op_set = tr_op_set
		self.train_summary_op = train_summary_op
		self.dev_summary_op = dev_summary_op
		self.train_summary_writer = train_summary_writer
		self.dev_summary_writer = dev_summary_writer


class InputTensors:
	def __init__(self, input_x1, input_x2, input_y, dropout_keep_prob):
		self.input_x1, self.input_x2, self.input_y = input_x1, input_x2, input_y
		self.dropout_keep_prob = dropout_keep_prob


class ResultTensors:
	def __init__(self, global_step, loss, accuracy, distance, temp_sim):
		self.global_step, self.loss, self.accuracy, self.distance, self.temp_sim = global_step, loss, accuracy, distance, temp_sim


class Trainer(NLPApp):
	def __init__(self, FLAGs):
		self.FLAGS = FLAGs
		self.inpH = InputHelper()
		self.session_conf = tf.ConfigProto(
			allow_soft_placement=self.FLAGS.allow_soft_placement,
			log_device_placement=self.FLAGS.log_device_placement)

	def __load_word2vec(self):

		trainableEmbeddings = False
		if self.FLAGS.is_char_based == True:
			self.FLAGS.word2vec_model = False
		else:
			if self.FLAGS.word2vec_model == None:
				trainableEmbeddings = True
				print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
				      "You are using word embedding based semantic similarity but "
				      "word2vec model path is empty. It is Recommended to use  --word2vec_model  argument. "
				      "Otherwise now the code is automatically trying to learn embedding values (may not help in accuracy)"
				      "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
			else:
				self.inpH.loadW2V(self.FLAGS.word2vec_model, self.FLAGS.word2vec_format)
		return trainableEmbeddings

	def __build_storage_path(self):
		'''
		Ex.
		out_dir:                        runs/1548973755
		checkpoint_dir_abs:             runs/1548973755/checkpoints
		checkpoint_model_abs:           runs/1548973755/checkpoints/model
		checkpoint_saved_model_abs:     runs/1548973755/checkpoints/model-XXX
		vocab_path:                     runs/1548973755/checkpoints/vocab
		:return:
		'''
		checkpoint_dir_abs = os.path.abspath(self.FLAGS.checkpoint_dir)

		# run/1412312455/checkpoints
		if self.FLAGS.checkpoint_dir and os.path.exists(checkpoint_dir_abs):
			# run/1412312455/
			print("Checkpoint dir:{} exists, loading vocab and weights from it".format(self.FLAGS.checkpoint_dir))
			out_dir = os.path.join(checkpoint_dir_abs, os.pardir)
		else:
			# Output directory for models and summaries
			timestamp = str(int(time.time()))
			out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
			checkpoint_dir_abs = os.path.abspath(os.path.join(out_dir, "checkpoints"))
			os.makedirs(checkpoint_dir_abs)

		checkpoint_model_abs = os.path.join(checkpoint_dir_abs, "model")

		print("Writing to {}\n".format(out_dir))
		checkpoint_saved_model_abs = os.path.join(checkpoint_dir_abs, self.FLAGS.model)
		vocab_path = os.path.join(checkpoint_dir_abs, "vocab")
		return out_dir, checkpoint_dir_abs, checkpoint_model_abs, checkpoint_saved_model_abs, vocab_path

	def run(self):
		'''
		Main logic of the app
		:return:
		'''
		# define all the path
		out_dir, checkpoint_dir_abs, checkpoint_model_abs, checkpoint_saved_model_abs, vocab_path = self.__build_storage_path()

		# splitting test and val data
		train_set, dev_set, vocab_processor, sum_no_of_batches = self.inpH.getDataSets(self.FLAGS.training_files, self.FLAGS.max_document_length, 10,
		                                                                               self.FLAGS.batch_size, self.FLAGS.is_char_based, vocab_path)

		# structure the model either from build or reload
		if self.FLAGS.model and os.path.exists("{}.meta".format(checkpoint_saved_model_abs)):
			print("loading trained model from check point:{}".format(checkpoint_saved_model_abs))
			saver, sess, input_tensors, result_tensors, metric_ops = self.__launch_from_load(checkpoint_saved_model_abs, out_dir)
		else:
			trainableEmbeddings = self.__load_word2vec()
			initW = self.__init_embedding_matrix(vocab_processor)
			saver, sess, input_tensors, result_tensors, metric_ops = self.__launch_from_build(vocab_processor, trainableEmbeddings, out_dir,
			                                                                                         checkpoint_dir_abs, initW)

		# train batches
		self.__run_batches(sess, sum_no_of_batches, train_set, dev_set, saver,
		                   input_tensors, result_tensors, metric_ops, checkpoint_model_abs)

		# don't forget to close the session
		sess.close()

	def __launch_from_load(self, model_path, out_dir):

		graph = tf.Graph()

		# this with is necessary, even you set graph para to init sess
		with graph.as_default():
			saver = tf.train.import_meta_graph("{}.meta".format(model_path))

		sess = tf.Session(graph=graph, config=self.session_conf)
		with sess.as_default():

			saver.restore(sess, model_path)

			# Get the placeholders from the graph by name
			input_x1 = graph.get_operation_by_name("input_x1").outputs[0]
			input_x2 = graph.get_operation_by_name("input_x2").outputs[0]
			input_y = graph.get_operation_by_name("input_y").outputs[0]

			dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

			global_step = graph.get_operation_by_name("global_step").outputs[0]
			loss = graph.get_operation_by_name("loss/loss_fun").outputs[0]
			accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]
			distance = graph.get_operation_by_name("output/distance").outputs[0]
			temp_sim = graph.get_operation_by_name("accuracy/temp_sim").outputs[0]

			# Tensors we want to evaluate
			tr_op_set = graph.get_operation_by_name("tr_op_set").outputs[0]
			train_summary_op = graph.get_operation_by_name("train_summary_op").outputs[0]
			dev_summary_op = graph.get_operation_by_name("dev_summary_op").outputs[0]

			train_summary_dir = os.path.join(out_dir, "summaries", "train")
			train_summary_writer = tf.summary.FileWriter(train_summary_dir, graph)
			dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
			dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, graph)

			input_tensors = InputTensors(input_x1, input_x2, input_y, dropout_keep_prob)
			result_tensors = ResultTensors(global_step, loss, accuracy, distance, temp_sim)
			metric_ops = MetricOps(tr_op_set, train_summary_op, dev_summary_op, train_summary_writer, dev_summary_writer)

		return saver, sess, input_tensors, result_tensors, metric_ops

	def __launch_from_build(self, vocab_processor, trainableEmbeddings, out_dir, checkpoint_dir_abs, initW):
		# ==================================================
		print("starting graph def")
		graph = tf.Graph()

		with graph.as_default():
			# will use default_graph as input para, and current default_graph is the `graph`
			sess = tf.Session(graph=graph, config=self.session_conf)
			print("started session")
			with sess.as_default():
				if self.FLAGS.is_char_based:
					siameseModel = SiameseLSTM(
						sequence_length=self.FLAGS.max_document_length,
						vocab_size=len(vocab_processor.vocabulary_),
						embedding_size=self.FLAGS.embedding_dim,
						hidden_units=self.FLAGS.hidden_units,
						l2_reg_lambda=self.FLAGS.l2_reg_lambda,
						batch_size=self.FLAGS.batch_size
					)
				else:
					siameseModel = SiameseLSTMw2v(
						sequence_length=self.FLAGS.max_document_length,
						vocab_size=len(vocab_processor.vocabulary_),
						embedding_size=self.FLAGS.embedding_dim,
						hidden_units=self.FLAGS.hidden_units,
						l2_reg_lambda=self.FLAGS.l2_reg_lambda,
						batch_size=self.FLAGS.batch_size,
						trainableEmbeddings=trainableEmbeddings
					)

			# Define Training procedure
			global_step = tf.Variable(0, name="global_step", trainable=False)
			optimizer = tf.train.AdamOptimizer(1e-3)
			print("initialized siameseModel object")

			grads_and_vars = optimizer.compute_gradients(siameseModel.loss)
			tr_op_set = optimizer.apply_gradients(grads_and_vars, global_step=global_step, name='tr_op_set')
			print("defined training_ops")
			# Keep track of gradient values and sparsity (optional)
			grad_summaries = []
			for g, v in grads_and_vars:
				if g is not None:
					grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
					sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
					grad_summaries.append(grad_hist_summary)
					grad_summaries.append(sparsity_summary)
			grad_summaries_merged = tf.summary.merge(grad_summaries)
			print("defined gradient summaries")

			# Summaries for loss and accuracy
			loss_summary = tf.summary.scalar("loss", siameseModel.loss)
			acc_summary = tf.summary.scalar("accuracy", siameseModel.accuracy)

			# Train Summaries
			train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
			train_summary_op = tf.identity(train_summary_op, 'train_summary_op')
			train_summary_dir = os.path.join(out_dir, "summaries", "train")
			train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

			# Dev summaries
			dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
			dev_summary_op = tf.identity(dev_summary_op, 'dev_summary_op')
			dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
			dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

			saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
			sess.run(tf.global_variables_initializer())
			if initW is not None:
				sess.run(siameseModel.W.assign(initW))

			graphpb_txt = str(graph.as_graph_def())
			with open(os.path.join(checkpoint_dir_abs, "graphpb.txt"), 'w') as f:
				f.write(graphpb_txt)

		input_tensors = InputTensors(siameseModel.input_x1, siameseModel.input_x2, siameseModel.input_y, siameseModel.dropout_keep_prob)
		result_tensors = ResultTensors(global_step, siameseModel.loss, siameseModel.accuracy, siameseModel.distance, siameseModel.temp_sim)
		metric_ops = MetricOps(tr_op_set, train_summary_op, dev_summary_op, train_summary_writer, dev_summary_writer)
		return saver, sess, input_tensors, result_tensors, metric_ops

	def __run_batches(self, sess, sum_no_of_batches, train_set, dev_set, saver, input_tensors, result_tensors, metric_ops, checkpoint_prefix):

		# Generate batches，Seq of [question1_tokenized, question2_tokenized, label]
		batches = self.inpH.batch_iter(list(zip(train_set[0], train_set[1], train_set[2])),
		                               self.FLAGS.batch_size, self.FLAGS.num_epochs)

		max_validation_acc = 0.0
		for nn in range(sum_no_of_batches * self.FLAGS.num_epochs):
			batch = next(batches)
			if len(batch) < 1:
				continue
			x1_batch, x2_batch, y_batch = zip(*batch)
			if len(y_batch) < 1:
				continue
			self.__train_step(sess, input_tensors, result_tensors, metric_ops, x1_batch, x2_batch, y_batch)
			current_step = tf.train.global_step(sess, result_tensors.global_step)
			sum_acc = 0.0
			if current_step % self.FLAGS.evaluate_every == 0:
				print("\nEvaluation:")
				dev_batches = self.inpH.batch_iter(list(zip(dev_set[0], dev_set[1], dev_set[2])), self.FLAGS.batch_size, 1)
				for db in dev_batches:
					if len(db) < 1:
						continue
					x1_dev_b, x2_dev_b, y_dev_b = zip(*db)
					if len(y_dev_b) < 1:
						continue

					acc = self.__dev_step(sess, input_tensors, result_tensors, metric_ops, x1_dev_b, x2_dev_b, y_dev_b)
					sum_acc = sum_acc + acc
				print("")

			# 如果当前模型在validation数据上精确度提高了，那么打印metric并保存模型
			if current_step % self.FLAGS.checkpoint_every == 0:
				if sum_acc >= max_validation_acc:
					max_validation_acc = sum_acc
					saver.save(sess, checkpoint_prefix, global_step=current_step)
					tf.train.write_graph(sess.graph.as_graph_def(), checkpoint_prefix, "graph" + str(nn) + ".pb", as_text=True)
					print("Saved model {} with sum_accuracy={} checkpoint to {}\n".format(nn, max_validation_acc, checkpoint_prefix))

	def __train_step(self, sess, input_tensors, result_tensors, metric_ops, x1_batch, x2_batch, y_batch):

		"""
		A single training step
		"""
		# 为什么要不时的颠倒输入的句子对的顺序？损失函数应该和输入顺序无关啊？
		if random() > 0.5:
			x1_batch, x2_batch = x2_batch, x1_batch

		feed_dict = {
			input_tensors.input_x1: x1_batch,
			input_tensors.input_x2: x2_batch,
			input_tensors.input_y: y_batch,
			input_tensors.dropout_keep_prob: 1.0,
		}
		_, step, loss, accuracy, dist, sim, summaries = sess.run(
			[metric_ops.tr_op_set, result_tensors.global_step, result_tensors.loss, result_tensors.accuracy, result_tensors.distance,
			 result_tensors.temp_sim, metric_ops.train_summary_op],
			feed_dict)

		time_str = datetime.datetime.now().isoformat()
		print("TRAIN {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
		metric_ops.train_summary_writer.add_summary(summaries, step)

	def __dev_step(self, sess, input_tensors, result_tensors, metric_ops, x1_batch, x2_batch, y_batch):
		"""
		A single training step
		"""
		if random() > 0.5:
			x1_batch, x2_batch = x2_batch, x1_batch

		feed_dict = {
			input_tensors.input_x1: x1_batch,
			input_tensors.input_x2: x2_batch,
			input_tensors.input_y: y_batch,
			input_tensors.dropout_keep_prob: 1.0,
		}
		step, loss, accuracy, sim, summaries = sess.run(
			[result_tensors.global_step, result_tensors.loss, result_tensors.accuracy, result_tensors.temp_sim, metric_ops.dev_summary_op],
			feed_dict)

		time_str = datetime.datetime.now().isoformat()
		print("DEV {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
		metric_ops.dev_summary_writer.add_summary(summaries, step)
		# print(y_batch, sim)
		return accuracy

	def __init_embedding_matrix(self, vocab_processor):

		if self.FLAGS.word2vec_model:
			# initial embedding matrix with random uniform
			initW = np.random.uniform(-0.25, 0.25, (len(vocab_processor.vocabulary_), self.FLAGS.embedding_dim))
			# initW = np.zeros(shape=(len(vocab_processor.vocabulary_), FLAGS.embedding_dim))
			# load any vectors from the word2vec
			print("initializing initW with pre-trained word2vec embeddings")
			for w in vocab_processor.vocabulary_._mapping:
				arr = []
				# 去掉词中所有非数字和字母的字符
				s = re.sub('[^0-9a-zA-Z]+', '', w)
				if w in self.inpH.pre_emb:
					arr = self.inpH.pre_emb[w]
				elif w.lower() in self.inpH.pre_emb:
					arr = self.inpH.pre_emb[w.lower()]
				elif s in self.inpH.pre_emb:
					arr = self.inpH.pre_emb[s]
				elif s.isdigit():
					arr = self.inpH.pre_emb["zero"]

				if len(arr) > 0:
					# sometime, the vector of the word may start with an offset, use the last embedding_dim numbers will solve the problem.
					if len(arr) > self.FLAGS.embedding_dim:
						arr = arr[-self.FLAGS.embedding_dim:]
					idx = vocab_processor.vocabulary_.get(w)
					initW[idx] = np.asarray(arr).astype(np.float32)

				# 如果arr是[]，那么代表数据中的词在trained word2vec中不存在，那么就用最开始随机的weights来训练

			print("Done assigning intiW. len=" + str(len(initW)))
			# initW 会作为新的embedding matrix在内存中运行， 把inpH中的PreEmb哈希表删除释放缓存！
			self.inpH.deletePreEmb()
			gc.collect()
			return initW


def arg_parser():
	tf.flags.DEFINE_boolean("is_char_based", False, "is character based syntactic similarity. "
	                                                "if false then word embedding based semantic similarity is used."
	                                                "(default: True)")

	tf.flags.DEFINE_string("word2vec_model", "data/wiki.simple.vec", "word2vec pre-trained embeddings file (default: None)")
	tf.flags.DEFINE_string("word2vec_format", "text", "word2vec pre-trained embeddings file format (bin/text/textgz)(default: None)")
	tf.flags.DEFINE_string("checkpoint_dir", "runs/1548976511/checkpoints", "Checkpoint directory from training run")
	tf.flags.DEFINE_string("model", "model-2000", "Load trained model checkpoint (Default: None)")

	tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 300)")
	tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "Dropout keep probability (default: 1.0)")
	tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
	# for sentence semantic similarity use
	tf.flags.DEFINE_string("training_files", "data/train_snli.txt", "training file (default: None)")
	#  "train_snli.txt"
	tf.flags.DEFINE_integer("hidden_units", 50, "Number of hidden units (default:50)")

	# Training parameters
	tf.flags.DEFINE_integer("max_document_length", 10, "Sequence size feeding into the RNN model")
	tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
	tf.flags.DEFINE_integer("num_epochs", 300, "Number of training epochs (default: 200)")
	tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps (default: 100)")
	tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
	# Misc Parameters
	tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
	tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

	FLAGS = tf.flags.FLAGS
	FLAGS._parse_flags()
	print("\nParameters:")
	for attr, value in sorted(FLAGS.__flags.items()):
		print("{}={}".format(attr.upper(), value))
	print("")

	if FLAGS.training_files == None:
		print("Input Files List is empty. use --training_files argument.")
		exit()

	return FLAGS


if __name__ == "__main__":
	FLAGs = arg_parser()
	app = Trainer(FLAGs)
	app.run()
