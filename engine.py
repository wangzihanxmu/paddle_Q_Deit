from quantization.lsq_layer import QuantAct, QuantConv2d, QuantLinear, QuantMultiHeadAct, QuantMuitiHeadLinear, QuantMuitiHeadLinear_in
import paddle
# paddle.disable_static()
@paddle.no_grad()
def initialize_quantization(data_loader, model, device, output_dir, sample_iters=5):

    # metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Initialization:'
    # if utils.is_main_process():
    with (output_dir / "scales.txt").open("w") as f:
        f.write("weight scales:\n")
        for name, m in model.named_sublayers():
            if (isinstance(m, QuantLinear) or isinstance(m, QuantConv2d) or isinstance(m, QuantMuitiHeadLinear) or isinstance(m, QuantMuitiHeadLinear_in)) and m.alpha is not None:
                print(f"initialize the weight scale for module {name}")
                m.initialize_scale(device)
                f.write(name + ': ' + str(m.alpha.numpy()) + '\n')

        # switch to evaluation mode
        model.eval()
        f.write("activation scales:\n")
        n = 0
        for images, target in data_loader:
            n += 1
            if n > sample_iters:
                break
            # images = images.to(device, non_blocking=True)

            # compute output
            # with paddle.cuda.amp.autocast():
            output = model(images)
        for name, m in model.named_sublayers():
            if (isinstance(m, QuantAct) or isinstance(m, QuantMultiHeadAct)) and m.alpha is not None:
                print(f"initialize the activation scale for module {name}")
                m.initialize_scale_offset(device)
                f.write(name + ': ' + str(m.alpha.numpy()) + '\n')
                if m.offset:
                    f.write("offset" + ': ' + str(m.beta.numpy()) + '\n')
    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()

    return

def initialize_muitihead_quantization(model, device):
    for name, m in model.named_sublayers():
        if (isinstance(m, QuantMuitiHeadLinear) or isinstance(m, QuantMuitiHeadLinear_in) or isinstance(m, QuantMultiHeadAct)) and m.alpha is not None:
            m.nbits = paddle.create_parameter(shape=(paddle.ones(m.num_head).to(device) * m.nbits).shape,
                                        dtype=str((paddle.ones(m.num_head).to(device) * m.nbits).numpy().dtype),
                                        default_initializer=paddle.nn.initializer.Assign(paddle.ones(m.num_head).to(device) * m.nbits))
            print(f"Initialize bit-width for {name}, bit:{m.nbits.data}")
# @paddle.no_grad()
# def update_bn(data_loader, model, device):
#     criterion = paddle.nn.CrossEntropyLoss()

#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = 'Update BN:'

#     # switch to evaluation mode
#     model.train()

#     for images, _ in metric_logger.log_every(data_loader, 10, header):
#         images = images.to(device, non_blocking=True)

#         # compute output
#         output = model(images)

#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()


#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# @paddle.no_grad()
# def log_quantization_parameters(model, output_dir):
#     with (output_dir / "q_param.txt").open("w") as f:
#         f.write("weight scales:\n")
#         for name, m in model.named_modules():
#             if isinstance(m, QuantLinear) or isinstance(m, QuantConv2d) or isinstance(m, QuantMultiHeadAct) or isinstance(m, QuantAct) or isinstance(m, QuantMuitiHeadLinear) or isinstance(m, QuantMuitiHeadLinear_in):
#                 if m.alpha is not None:
#                     f.write(name + '\n')
#                     f.write('--bitwidth: ' + str(m.nbits.data) + '\n')
#                     if m.nbits.grad is not None:
#                         f.write('--grad: ' + str(m.nbits.grad.data) + '\n')
#                     n = m.nbits.round().to(paddle.long)
#                     f.write('--scales: ' + str(m.alpha.data) + '\n')
#                     f.write('--scale_n: ' + str(m.alpha[n-2]) + '\n')
#                     if m.alpha.grad is not None:
#                         f.write('--grad: ' + str(m.alpha.grad.data) + '\n')
#                 if isinstance(m, QuantAct) and m.offset:
#                     f.write("--offsets: " + str(m.beta.data) + '\n')
#                     f.write("--offset_n: " + str(m.beta[n-2]) + '\n')
