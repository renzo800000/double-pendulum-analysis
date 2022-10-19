function [f] = base_functions_library()
% Return a list of base functions for the double pendulum system
%
% returns a list of base functions that can be used to solve the Double
% Pendulum system with a SINDy algorithm.
%
% It will return the basis functions as symbolic expressions, so that 
% we can label the final result and obtain an expression for the final
% ODEs.

% Create symbolic variables for  theta1, omega1, theta2, omega2
t1 = sym('t1');
o1 = sym('o1');
t2 = sym('t2');
o2 = sym('o2');

% Create a list of polynomial base functions
f_poly = [ 1, t1, t2, o1, o2, t1.^2, t2.^2, o1.^2, o2.^2];

% Create a list of trigonometric base functions
f_trig = [1];
f_trig = [f_trig, sin(t1), cos(t1), sin(t1).^2, sin(t1).*cos(t1), cos(t1).^2];
f_trig = [f_trig, sin(t2), cos(t2), sin(t2).^2, sin(t2).*cos(t2), cos(t2).^2];
f_trig = [f_trig, sin(t1-t2), cos(t1-t2), sin(t1-t2).^2, sin(t1-t2).*cos(t1-t2), cos(t1-t2).^2];
f_trig = [f_trig, sin(t2-t1), cos(t2-t1), sin(t2-t1).^2, sin(t2-t1).*cos(t2-t1), cos(t2-t1).^2];
f_trig = [f_trig, sin(t1-2*t2), cos(t1-2*t2), sin(t1-2*t2).^2, sin(t1-2*t2).*cos(t1-2*t2), cos(t1-2*t2).^2];
f_trig = [f_trig, sin(t2-2*t1), cos(t2-2*t1), sin(t2-2*t1).^2, sin(t2-2*t1).*cos(t2-2*t1), cos(t2-2*t1).^2];
f_trig = [f_trig, sin(2*(t1-t2)), cos(2*(t1-t2)), sin(2*(t1-t2)).^2, sin(2*(t1-t2)).*cos(2*(t1-t2)), cos(2*(t1-t2)).^2];

% Combine every polynomial base function with every trigonometric base
% function (do a matrix multiplication, then flatten the result)
f_mat = (f_poly.')*f_trig;
f = reshape(f_mat,1,[]);

end
