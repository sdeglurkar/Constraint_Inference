classdef Node
    properties
        x
        y
        theta
        cost
        pind    %Pointer to parent
        tstamp
    end
    
    methods
        
        function obj = Node(x, y, theta, cost, pind, tstamp)
            obj.x = x;
            obj.y = y;
            obj.theta = theta;
            obj.cost = cost;
            obj.pind = pind;
            obj.tstamp = tstamp;
        end
        
        function value = hash(obj, xmin, ymin, thetamin, xwidth, thetawidth)
            value = ((obj.y - ymin) * xwidth + (obj.x - xmin)) * thetawidth + (obj.theta - thetamin);
            value = num2str(value);
        end
        
    end
end


