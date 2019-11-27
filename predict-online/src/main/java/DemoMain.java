import com.google.common.collect.Lists;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import org.tensorflow.framework.*;
import tensorflow.serving.*;

import java.util.*;

/**
 * Created by pfliu on 2019/11/26.
 */
public class DemoMain {


    public static void main(String[] args) {
        // 构造请求
        ManagedChannel channel = ManagedChannelBuilder.forAddress("192.168.1.xxx", 4590).usePlaintext(true).build();
        PredictionServiceGrpc.PredictionServiceBlockingStub stub = PredictionServiceGrpc.newBlockingStub(channel);
        Predict.PredictRequest.Builder predictRequestBuilder = Predict.PredictRequest.newBuilder();
        Model.ModelSpec.Builder modelSpecBuilder = Model.ModelSpec.newBuilder();
        // 你的模型的名字
        modelSpecBuilder.setName("model");
        modelSpecBuilder.setSignatureName("");
        predictRequestBuilder.setModelSpec(modelSpecBuilder);
        TensorProto.Builder tensorProtoBuilder = TensorProto.newBuilder();
        // 模型接受的数据类型
        tensorProtoBuilder.setDtype(DataType.DT_FLOAT);
        TensorShapeProto.Builder tensorShapeBuilder = TensorShapeProto.newBuilder();
        // 接受数据的shape, 几维的数组, 每一维多少个. 我的测试数据是三个.
        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(1));
        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(3));

        // 我的测试数据, 这里需要把输入的字符串进行编码. 比如在我的编码下, 比如将 : 呼延十 编码成下面三个数字.
        // 注: 这里并不是真实编码, 可以查看train-offline/data/vocab.json 以查看编码方式.
        String s = "呼延十";
        List<Float> ret = new ArrayList<>();
        ret.add(348.0f);
        ret.add(3848.0f);
        ret.add(2557.0f);

        tensorProtoBuilder.setTensorShape(tensorShapeBuilder.build());
        tensorProtoBuilder.addAllFloatVal(ret);
        predictRequestBuilder.putInputs("input", tensorProtoBuilder.build());
        Predict.PredictResponse predictResponse = stub.predict(predictRequestBuilder.build());

        //  这里拿到的是一个(1,1,3)的矩阵. 所以我们需要把他解码成我们想要的tag. 涉及到你的tag列表.
        List<Float> output = predictResponse.getOutputsMap().get("output").getFloatValList();
        List<String> tags = Arrays.asList("O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "i-ORG");
        List<String> rets = phraseFrom(s, output, tags);
        System.out.println(rets);

    }


    private static List<String> phraseFrom(String q, List<Float> output, List<String> tags) {
        List<List<Float>> partition = Lists.partition(output, tags.size());
        List<Integer> idx = new ArrayList<>();
        for (List<Float> floats : partition) {
            for (int j = 0; j < floats.size(); j++) {
                if (floats.get(j) == 1.0f) {
                    idx.add(j);
                    break;
                }
            }
        }
        assert q.length() != idx.size();
        // 从query和每个字的tag解析成词语的意图.
        StringBuilder sb = new StringBuilder();
        char[] chars = q.toCharArray();
        List<String> rets = new ArrayList<>();
        for (int i = 0; i < chars.length; i++) {
            Integer tag = idx.get(i);
            if ((tag & 1) == 1 && sb.length() != 0) {
                String item = sb.toString();
                String ret = tags.get(idx.get(i - 1));
                rets.add(ret);
                sb.setLength(0);
                sb.append(chars[i]);
            } else {
                sb.append(chars[i]);
            }
        }
        if (sb.length() != 0) {
            String ret = tags.get(idx.get(q.length() - 1));
            rets.add(ret);
        }
        return rets;
    }

}
