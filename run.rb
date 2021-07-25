(1..5).step(1) do |x|
    %W(mnist_lenet_300_100 cifar_vgg_16 cifar_resnet_20).each do |a|
        [false, true].each do |is_local|
            cmd = "python open_lth.py lottery --default_hparams=#{a} --levels=15 --apex_fp16 --replicate=#{x}"
            if is_local
                cmd += ' --pruning_strategy=sparse_local'
            end
            puts cmd
            `#{cmd}`
        end
    end
end