% Copyright 2026 The MathWorks, Inc.
%#codegen

classdef ReplaceClsEmbeddingLayer < nnet.layer.Layer & nnet.layer.Acceleratable

    methods
        function layer = ReplaceClsEmbeddingLayer(params)

            arguments
                params.Name (1, 1) string = "ReplaceClsEmbeddingLayer"
            end

            layer.Name = params.Name;
            layer.Description = "Replace CLS embedding token.";
            layer.NumInputs = 2;
            layer.InputNames = {'source', 'dest'};
        end

        function Z = predict(~, source, dest)
            % Inputs are in CBT format
            Z = dest;
            Z(:, :, 1) = source(:, :, 1);
        end

    end
end