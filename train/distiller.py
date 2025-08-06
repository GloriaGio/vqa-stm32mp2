import tensorflow as tf
from tensorflow import keras


class Distiller(keras.Model):

    def __init__(self, student):
        super().__init__()
        self.student = student
        self.distill_loss_tracker = keras.metrics.Mean(name="distillation_loss")
        self.student_loss_tracker = keras.metrics.Mean(name="student_loss")

    @property
    def metrics(self):
        metrics = super().metrics
        metrics.append(self.distill_loss_tracker)
        metrics.append(self.student_loss_tracker)
        return metrics

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # unpack data
        x, y, w = data
        y, logits = y

        teacher_logits = logits

        with tf.GradientTape() as tape:
            student_logits = self.student(x, training=True)

            # compute losses
            student_loss = (
                self.student_loss_fn(y, tf.nn.softmax(student_logits, axis=1)) * w
            )
            distillation_loss = (
                self.distillation_loss_fn(
                    tf.nn.softmax(teacher_logits / self.temperature, axis=1),
                    tf.nn.softmax(student_logits / self.temperature, axis=1),
                )
                * (self.temperature**2)
                * w
            )
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # update metrics
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            elif metric.name == "distillation_loss":
                metric.update_state(distillation_loss)
            elif metric.name == "student_loss":
                metric.update_state(student_loss)
            else:
                metric.update_state(y, tf.nn.softmax(student_logits, axis=1))

        # return performance
        results = {m.name: m.result() for m in self.metrics}
        return results

    def test_step(self, data):
        x, y, w = data
        y, logits = y

        teacher_logits = logits
        student_logits = self.student(x, training=False)

        student_loss = (
            self.student_loss_fn(y, tf.nn.softmax(student_logits, axis=1)) * w
        )
        distillation_loss = (
            self.distillation_loss_fn(
                tf.nn.softmax(teacher_logits / self.temperature, axis=1),
                tf.nn.softmax(student_logits / self.temperature, axis=1),
            )
            * (self.temperature**2)
            * w
        )
        loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            elif metric.name == "distillation_loss":
                metric.update_state(distillation_loss)
            elif metric.name == "student_loss":
                metric.update_state(student_loss)
            else:
                metric.update_state(y, tf.nn.softmax(student_logits, axis=1))

        results = {m.name: m.result() for m in self.metrics}
        return results

    def call(self, x):
        return self.student(x, training=False)
