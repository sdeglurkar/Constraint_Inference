classdef Dummy
    properties
        dummy_var
    end
    
    methods
        
        function obj = Dummy(dummy_var)
            obj.dummy_var = dummy_var;
        end
        
        function lulu = doSmt(obj)
            lulu = obj.dummy_var * 500;
        end
    end
    
end