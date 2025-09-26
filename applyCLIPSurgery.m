% Copyright 2026 The MathWorks, Inc.

function imageEncoder = applyCLIPSurgery(clipImageEncoder)
    arguments
        clipImageEncoder (1, 1) dlnetwork
    end
    
    imageEncoder = clipImageEncoder; % Do not modify in place.
    imageEncoder = addDualPath(imageEncoder);
    imageEncoder = initialize(imageEncoder);
end


function imageEncoder = addDualPath(imageEncoder)
    % Adds the dual path, reusing parameters from existing selfAttention layers.
    % This involves two steps:
    % 1. Replace the existing selfAttentionLayers with attentionLayers.
    %    This requires a separate fullyConnectedLayer for each of Q, K, V,
    %    and Output projection.
    % 2. Add the dual path, reusing the output from the attentionLayers in
    %    step 1.
    
    replaceDepth = 6; % The number of transformer blocks from the end to modify.
    numBlocks = 24; % The number of transformer blocks in the model.
    blockIdxs = numBlocks-replaceDepth+1:numBlocks;
    
    % The attention layers are 0-indexed, everything else is 1-indexed.
    layerNames = "transformer:TopLevelModule:CLIPModel_visual:transformer:resblocks:" ...
        + string(blockIdxs - 1) + ":attn:SelfAttention";
    
    for idx = 1:numel(layerNames)
        blockNum = blockIdxs(idx);
        disp("Modifying transformer " + blockNum)

        % ---------------------------- Step 1 ----------------------------

        % Get the name/layer of the selfAttention layer in the normal path,
        % and its input/output layers.
        layerName = layerNames(idx);
        layer = getLayer(imageEncoder, layerName);
        imageEncoder = removeLayers(imageEncoder, layerName);
        inputName = "transformer:resblock" + num2str(blockNum) + "_ln1";
        outputName = "transformer:resblock" + num2str(blockNum) + "_add1";

        % Replace the selfAttention layer with an attentionLayer, so the
        % Value paramaters are reusable.
        newLayerName = "ClipAttention"+blockNum;
        newAttnLayer = attentionLayer(layer.NumHeads, Name=newLayerName);
        imageEncoder = addLayers(imageEncoder, newAttnLayer);

        % Set up the QKV inputs as fully connected layers, and copy their
        % parameters.  Connect their input and output.
        newQueryLayer = copyFCLayer(layer, newLayerName+"query", FromAttention="Query");
        imageEncoder = addLayers(imageEncoder, newQueryLayer);
        imageEncoder = connectLayers(imageEncoder, inputName, newQueryLayer.Name);
        imageEncoder = connectLayers(imageEncoder, newQueryLayer.Name, newAttnLayer.Name + "/query");

        newKeyLayer = copyFCLayer(layer, newLayerName+"key", FromAttention="Key");
        imageEncoder = addLayers(imageEncoder, newKeyLayer);
        imageEncoder = connectLayers(imageEncoder, inputName, newKeyLayer.Name);
        imageEncoder = connectLayers(imageEncoder, newKeyLayer.Name, newAttnLayer.Name + "/key");

        newValueLayer = copyFCLayer(layer, newLayerName+"value", FromAttention="Value");
        imageEncoder = addLayers(imageEncoder, newValueLayer);
        imageEncoder = connectLayers(imageEncoder, inputName, newValueLayer.Name);
        imageEncoder = connectLayers(imageEncoder, newValueLayer.Name, newAttnLayer.Name + "/value");
        
        % Create and connect a fully connected layer for the output
        % projection of the attentionLayer.
        newOutputLayer = copyFCLayer(layer, newLayerName+"output", OutputLearnables=true, FromAttention="Output");
        imageEncoder = addLayers(imageEncoder, newOutputLayer);
        imageEncoder = connectLayers(imageEncoder, newAttnLayer.Name, newOutputLayer.Name);
        imageEncoder = connectLayers(imageEncoder, newOutputLayer.Name+"/out", outputName+"/in2");

        % ---------------------------- Step 2 ----------------------------

        % Construct a VVV attention layer from the normal path layer.
        vvvAttn = attentionLayer(layer.NumHeads, Name="vvv_attn"+blockNum);
        imageEncoder = addLayers(imageEncoder, vvvAttn);

        % VVV attention.
        imageEncoder = connectLayers(imageEncoder, newValueLayer.Name, vvvAttn.Name+"/query");
        imageEncoder = connectLayers(imageEncoder, newValueLayer.Name, vvvAttn.Name+"/value");
        imageEncoder = connectLayers(imageEncoder, newValueLayer.Name, vvvAttn.Name+"/key");

        vvvOutput = fullyConnectedLayer(newOutputLayer.OutputSize, InputLearnables=["bias","weights"], ...
            Name="vvvOutput" + blockNum);
        imageEncoder = addLayers(imageEncoder, vvvOutput);

        % Share output projection weights of QKV attention branch with VVV
        % attention branch.
        imageEncoder = connectLayers(imageEncoder, newOutputLayer.Name + "/weights", vvvOutput.Name + "/weights");
        imageEncoder = connectLayers(imageEncoder, newOutputLayer.Name + "/bias", vvvOutput.Name + "/bias");
    
        imageEncoder = connectLayers(imageEncoder, vvvAttn.Name, vvvOutput.Name + "/in");

        % Create an additionLayer for the output of the VVV attention layer.
        outputToVVVLayerName = "transformer:resblock" +  blockNum + "_dualpath_add3";
        outputLayer = additionLayer(2, Name=outputToVVVLayerName);
        imageEncoder = addLayers(imageEncoder, outputLayer);
    
        if idx == 1
            firstAddLayer = "transformer:resblock" + string(blockNum - 1) + "_add2";
        else
            previousName = "transformer:resblock" +  string(blockNum - 1) + "_dualpath_add3";
            firstAddLayer = getLayer(imageEncoder, previousName).Name;
        end
    
        % Connect the VVV attention layer and previous output to the additionLayer.
        add1Name = outputLayer.Name + "/in1";
        imageEncoder = connectLayers(imageEncoder, firstAddLayer, add1Name);
        add2Name = outputLayer.Name + "/in2";
        imageEncoder = connectLayers(imageEncoder, vvvOutput.Name, add2Name);
    end
    
    % Add layer from the dual path to the outputs of the image encoder.
    outputIdx = 24;
    outputName = "transformer:resblock" + outputIdx + "_dualpath_add3";
    
    % Set the CLS token embedding to come from the original path, and
    % remove the slice layer in the original path.
    imageEncoder = postProcessOutput(imageEncoder, outputName);
end

function imageEncoder = postProcessOutput(imageEncoder, outputName)

    layerNormLayer = getLayer(imageEncoder, "lnPost");
    origPathOutput = getLayer(imageEncoder, "transformer:resblock24_add2");

    replaceCLSEmbedLayer = ReplaceClsEmbeddingLayer(Name="ReplaceClsEmbeddingLayer");
    imageEncoder = addLayers(imageEncoder, replaceCLSEmbedLayer);

    imageEncoder = connectLayers(imageEncoder, outputName, replaceCLSEmbedLayer.Name + "/dest");
    imageEncoder = connectLayers(imageEncoder, origPathOutput.Name, replaceCLSEmbedLayer.Name + "/source");

    imageEncoder = removeLayers(imageEncoder, "slice");
    imageEncoder = connectLayers(imageEncoder, replaceCLSEmbedLayer.Name, layerNormLayer.Name);
end

function newLayer = copyFCLayer(oldLayer, newLayerName, params)
    arguments
        oldLayer
        newLayerName (1, 1) string
        params.OutputLearnables (1, 1) logical = false
        params.FromAttention (1, 1) string {mustBeMember(params.FromAttention, ["", "Query", "Key", "Value", "Output"])} = ""
    end

    if params.OutputLearnables
        newLayer = fullyConnectedLayer(oldLayer.OutputSize, Name=newLayerName, ...
            OutputLearnables=["weights", "bias"]);
    else
        newLayer = fullyConnectedLayer(oldLayer.OutputSize, Name=newLayerName);
    end

    newLayer.Weights = oldLayer.(params.FromAttention + "Weights");
    newLayer.Bias = oldLayer.(params.FromAttention + "Bias");

    newLayer.WeightsInitializer = oldLayer.WeightsInitializer;
    newLayer.WeightLearnRateFactor = oldLayer.WeightLearnRateFactor;
    newLayer.WeightL2Factor = oldLayer.WeightL2Factor;
    newLayer.BiasInitializer = oldLayer.BiasInitializer;
    newLayer.BiasLearnRateFactor = oldLayer.BiasLearnRateFactor;
    newLayer.BiasL2Factor = oldLayer.BiasL2Factor;
end