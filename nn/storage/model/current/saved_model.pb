ß­1
Ü¬
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
÷
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
°
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8·½.

!Adam/rnn_model_21/dense_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/rnn_model_21/dense_21/bias/v

5Adam/rnn_model_21/dense_21/bias/v/Read/ReadVariableOpReadVariableOp!Adam/rnn_model_21/dense_21/bias/v*
_output_shapes
:*
dtype0
¢
#Adam/rnn_model_21/dense_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *4
shared_name%#Adam/rnn_model_21/dense_21/kernel/v

7Adam/rnn_model_21/dense_21/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/rnn_model_21/dense_21/kernel/v*
_output_shapes

: *
dtype0
µ
.Adam/rnn_model_21/lstm_47/lstm_cell_185/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.Adam/rnn_model_21/lstm_47/lstm_cell_185/bias/v
®
BAdam/rnn_model_21/lstm_47/lstm_cell_185/bias/v/Read/ReadVariableOpReadVariableOp.Adam/rnn_model_21/lstm_47/lstm_cell_185/bias/v*
_output_shapes	
:*
dtype0
Ñ
:Adam/rnn_model_21/lstm_47/lstm_cell_185/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *K
shared_name<:Adam/rnn_model_21/lstm_47/lstm_cell_185/recurrent_kernel/v
Ê
NAdam/rnn_model_21/lstm_47/lstm_cell_185/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp:Adam/rnn_model_21/lstm_47/lstm_cell_185/recurrent_kernel/v*
_output_shapes
:	 *
dtype0
½
0Adam/rnn_model_21/lstm_47/lstm_cell_185/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *A
shared_name20Adam/rnn_model_21/lstm_47/lstm_cell_185/kernel/v
¶
DAdam/rnn_model_21/lstm_47/lstm_cell_185/kernel/v/Read/ReadVariableOpReadVariableOp0Adam/rnn_model_21/lstm_47/lstm_cell_185/kernel/v*
_output_shapes
:	 *
dtype0
µ
.Adam/rnn_model_21/lstm_46/lstm_cell_184/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.Adam/rnn_model_21/lstm_46/lstm_cell_184/bias/v
®
BAdam/rnn_model_21/lstm_46/lstm_cell_184/bias/v/Read/ReadVariableOpReadVariableOp.Adam/rnn_model_21/lstm_46/lstm_cell_184/bias/v*
_output_shapes	
:*
dtype0
Ñ
:Adam/rnn_model_21/lstm_46/lstm_cell_184/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *K
shared_name<:Adam/rnn_model_21/lstm_46/lstm_cell_184/recurrent_kernel/v
Ê
NAdam/rnn_model_21/lstm_46/lstm_cell_184/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp:Adam/rnn_model_21/lstm_46/lstm_cell_184/recurrent_kernel/v*
_output_shapes
:	 *
dtype0
½
0Adam/rnn_model_21/lstm_46/lstm_cell_184/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *A
shared_name20Adam/rnn_model_21/lstm_46/lstm_cell_184/kernel/v
¶
DAdam/rnn_model_21/lstm_46/lstm_cell_184/kernel/v/Read/ReadVariableOpReadVariableOp0Adam/rnn_model_21/lstm_46/lstm_cell_184/kernel/v*
_output_shapes
:	 *
dtype0
µ
.Adam/rnn_model_21/lstm_45/lstm_cell_183/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.Adam/rnn_model_21/lstm_45/lstm_cell_183/bias/v
®
BAdam/rnn_model_21/lstm_45/lstm_cell_183/bias/v/Read/ReadVariableOpReadVariableOp.Adam/rnn_model_21/lstm_45/lstm_cell_183/bias/v*
_output_shapes	
:*
dtype0
Ñ
:Adam/rnn_model_21/lstm_45/lstm_cell_183/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *K
shared_name<:Adam/rnn_model_21/lstm_45/lstm_cell_183/recurrent_kernel/v
Ê
NAdam/rnn_model_21/lstm_45/lstm_cell_183/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp:Adam/rnn_model_21/lstm_45/lstm_cell_183/recurrent_kernel/v*
_output_shapes
:	 *
dtype0
½
0Adam/rnn_model_21/lstm_45/lstm_cell_183/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*A
shared_name20Adam/rnn_model_21/lstm_45/lstm_cell_183/kernel/v
¶
DAdam/rnn_model_21/lstm_45/lstm_cell_183/kernel/v/Read/ReadVariableOpReadVariableOp0Adam/rnn_model_21/lstm_45/lstm_cell_183/kernel/v*
_output_shapes
:	*
dtype0

!Adam/rnn_model_21/dense_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/rnn_model_21/dense_21/bias/m

5Adam/rnn_model_21/dense_21/bias/m/Read/ReadVariableOpReadVariableOp!Adam/rnn_model_21/dense_21/bias/m*
_output_shapes
:*
dtype0
¢
#Adam/rnn_model_21/dense_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *4
shared_name%#Adam/rnn_model_21/dense_21/kernel/m

7Adam/rnn_model_21/dense_21/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/rnn_model_21/dense_21/kernel/m*
_output_shapes

: *
dtype0
µ
.Adam/rnn_model_21/lstm_47/lstm_cell_185/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.Adam/rnn_model_21/lstm_47/lstm_cell_185/bias/m
®
BAdam/rnn_model_21/lstm_47/lstm_cell_185/bias/m/Read/ReadVariableOpReadVariableOp.Adam/rnn_model_21/lstm_47/lstm_cell_185/bias/m*
_output_shapes	
:*
dtype0
Ñ
:Adam/rnn_model_21/lstm_47/lstm_cell_185/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *K
shared_name<:Adam/rnn_model_21/lstm_47/lstm_cell_185/recurrent_kernel/m
Ê
NAdam/rnn_model_21/lstm_47/lstm_cell_185/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp:Adam/rnn_model_21/lstm_47/lstm_cell_185/recurrent_kernel/m*
_output_shapes
:	 *
dtype0
½
0Adam/rnn_model_21/lstm_47/lstm_cell_185/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *A
shared_name20Adam/rnn_model_21/lstm_47/lstm_cell_185/kernel/m
¶
DAdam/rnn_model_21/lstm_47/lstm_cell_185/kernel/m/Read/ReadVariableOpReadVariableOp0Adam/rnn_model_21/lstm_47/lstm_cell_185/kernel/m*
_output_shapes
:	 *
dtype0
µ
.Adam/rnn_model_21/lstm_46/lstm_cell_184/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.Adam/rnn_model_21/lstm_46/lstm_cell_184/bias/m
®
BAdam/rnn_model_21/lstm_46/lstm_cell_184/bias/m/Read/ReadVariableOpReadVariableOp.Adam/rnn_model_21/lstm_46/lstm_cell_184/bias/m*
_output_shapes	
:*
dtype0
Ñ
:Adam/rnn_model_21/lstm_46/lstm_cell_184/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *K
shared_name<:Adam/rnn_model_21/lstm_46/lstm_cell_184/recurrent_kernel/m
Ê
NAdam/rnn_model_21/lstm_46/lstm_cell_184/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp:Adam/rnn_model_21/lstm_46/lstm_cell_184/recurrent_kernel/m*
_output_shapes
:	 *
dtype0
½
0Adam/rnn_model_21/lstm_46/lstm_cell_184/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *A
shared_name20Adam/rnn_model_21/lstm_46/lstm_cell_184/kernel/m
¶
DAdam/rnn_model_21/lstm_46/lstm_cell_184/kernel/m/Read/ReadVariableOpReadVariableOp0Adam/rnn_model_21/lstm_46/lstm_cell_184/kernel/m*
_output_shapes
:	 *
dtype0
µ
.Adam/rnn_model_21/lstm_45/lstm_cell_183/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.Adam/rnn_model_21/lstm_45/lstm_cell_183/bias/m
®
BAdam/rnn_model_21/lstm_45/lstm_cell_183/bias/m/Read/ReadVariableOpReadVariableOp.Adam/rnn_model_21/lstm_45/lstm_cell_183/bias/m*
_output_shapes	
:*
dtype0
Ñ
:Adam/rnn_model_21/lstm_45/lstm_cell_183/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *K
shared_name<:Adam/rnn_model_21/lstm_45/lstm_cell_183/recurrent_kernel/m
Ê
NAdam/rnn_model_21/lstm_45/lstm_cell_183/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp:Adam/rnn_model_21/lstm_45/lstm_cell_183/recurrent_kernel/m*
_output_shapes
:	 *
dtype0
½
0Adam/rnn_model_21/lstm_45/lstm_cell_183/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*A
shared_name20Adam/rnn_model_21/lstm_45/lstm_cell_183/kernel/m
¶
DAdam/rnn_model_21/lstm_45/lstm_cell_183/kernel/m/Read/ReadVariableOpReadVariableOp0Adam/rnn_model_21/lstm_45/lstm_cell_183/kernel/m*
_output_shapes
:	*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	

rnn_model_21/dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namernn_model_21/dense_21/bias

.rnn_model_21/dense_21/bias/Read/ReadVariableOpReadVariableOprnn_model_21/dense_21/bias*
_output_shapes
:*
dtype0

rnn_model_21/dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *-
shared_namernn_model_21/dense_21/kernel

0rnn_model_21/dense_21/kernel/Read/ReadVariableOpReadVariableOprnn_model_21/dense_21/kernel*
_output_shapes

: *
dtype0
§
'rnn_model_21/lstm_47/lstm_cell_185/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'rnn_model_21/lstm_47/lstm_cell_185/bias
 
;rnn_model_21/lstm_47/lstm_cell_185/bias/Read/ReadVariableOpReadVariableOp'rnn_model_21/lstm_47/lstm_cell_185/bias*
_output_shapes	
:*
dtype0
Ã
3rnn_model_21/lstm_47/lstm_cell_185/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *D
shared_name53rnn_model_21/lstm_47/lstm_cell_185/recurrent_kernel
¼
Grnn_model_21/lstm_47/lstm_cell_185/recurrent_kernel/Read/ReadVariableOpReadVariableOp3rnn_model_21/lstm_47/lstm_cell_185/recurrent_kernel*
_output_shapes
:	 *
dtype0
¯
)rnn_model_21/lstm_47/lstm_cell_185/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *:
shared_name+)rnn_model_21/lstm_47/lstm_cell_185/kernel
¨
=rnn_model_21/lstm_47/lstm_cell_185/kernel/Read/ReadVariableOpReadVariableOp)rnn_model_21/lstm_47/lstm_cell_185/kernel*
_output_shapes
:	 *
dtype0
§
'rnn_model_21/lstm_46/lstm_cell_184/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'rnn_model_21/lstm_46/lstm_cell_184/bias
 
;rnn_model_21/lstm_46/lstm_cell_184/bias/Read/ReadVariableOpReadVariableOp'rnn_model_21/lstm_46/lstm_cell_184/bias*
_output_shapes	
:*
dtype0
Ã
3rnn_model_21/lstm_46/lstm_cell_184/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *D
shared_name53rnn_model_21/lstm_46/lstm_cell_184/recurrent_kernel
¼
Grnn_model_21/lstm_46/lstm_cell_184/recurrent_kernel/Read/ReadVariableOpReadVariableOp3rnn_model_21/lstm_46/lstm_cell_184/recurrent_kernel*
_output_shapes
:	 *
dtype0
¯
)rnn_model_21/lstm_46/lstm_cell_184/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *:
shared_name+)rnn_model_21/lstm_46/lstm_cell_184/kernel
¨
=rnn_model_21/lstm_46/lstm_cell_184/kernel/Read/ReadVariableOpReadVariableOp)rnn_model_21/lstm_46/lstm_cell_184/kernel*
_output_shapes
:	 *
dtype0
§
'rnn_model_21/lstm_45/lstm_cell_183/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'rnn_model_21/lstm_45/lstm_cell_183/bias
 
;rnn_model_21/lstm_45/lstm_cell_183/bias/Read/ReadVariableOpReadVariableOp'rnn_model_21/lstm_45/lstm_cell_183/bias*
_output_shapes	
:*
dtype0
Ã
3rnn_model_21/lstm_45/lstm_cell_183/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *D
shared_name53rnn_model_21/lstm_45/lstm_cell_183/recurrent_kernel
¼
Grnn_model_21/lstm_45/lstm_cell_183/recurrent_kernel/Read/ReadVariableOpReadVariableOp3rnn_model_21/lstm_45/lstm_cell_183/recurrent_kernel*
_output_shapes
:	 *
dtype0
¯
)rnn_model_21/lstm_45/lstm_cell_183/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*:
shared_name+)rnn_model_21/lstm_45/lstm_cell_183/kernel
¨
=rnn_model_21/lstm_45/lstm_cell_183/kernel/Read/ReadVariableOpReadVariableOp)rnn_model_21/lstm_45/lstm_cell_183/kernel*
_output_shapes
:	*
dtype0

serving_default_input_1Placeholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ

¢
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1)rnn_model_21/lstm_45/lstm_cell_183/kernel3rnn_model_21/lstm_45/lstm_cell_183/recurrent_kernel'rnn_model_21/lstm_45/lstm_cell_183/bias)rnn_model_21/lstm_46/lstm_cell_184/kernel3rnn_model_21/lstm_46/lstm_cell_184/recurrent_kernel'rnn_model_21/lstm_46/lstm_cell_184/bias)rnn_model_21/lstm_47/lstm_cell_185/kernel3rnn_model_21/lstm_47/lstm_cell_185/recurrent_kernel'rnn_model_21/lstm_47/lstm_cell_185/biasrnn_model_21/dense_21/kernelrnn_model_21/dense_21/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_3533430

NoOpNoOp
ë]
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¦]
value]B] B]
Ú
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
m_layers
		optimizer


signatures*
R
0
1
2
3
4
5
6
7
8
9
10*
R
0
1
2
3
4
5
6
7
8
9
10*
* 
°
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
trace_0
trace_1
trace_2
trace_3* 
6
trace_0
 trace_1
!trace_2
"trace_3* 
* 
'
#0
$1
%2
&3
'4*
 
(iter

)beta_1

*beta_2
	+decay
,learning_ratemËmÌmÍmÎmÏmÐmÑmÒmÓmÔmÕvÖv×vØvÙvÚvÛvÜvÝvÞvßvà*

-serving_default* 
ic
VARIABLE_VALUE)rnn_model_21/lstm_45/lstm_cell_183/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE3rnn_model_21/lstm_45/lstm_cell_183/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'rnn_model_21/lstm_45/lstm_cell_183/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE)rnn_model_21/lstm_46/lstm_cell_184/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE3rnn_model_21/lstm_46/lstm_cell_184/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'rnn_model_21/lstm_46/lstm_cell_184/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE)rnn_model_21/lstm_47/lstm_cell_185/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE3rnn_model_21/lstm_47/lstm_cell_185/recurrent_kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'rnn_model_21/lstm_47/lstm_cell_185/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUErnn_model_21/dense_21/kernel&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUErnn_model_21/dense_21/bias'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
#0
$1
%2
&3
'4*

.0
/1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
Á
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6_random_generator
7cell
8
state_spec*
Á
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses
?_random_generator
@cell
A
state_spec*
Á
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses
H_random_generator
Icell
J
state_spec*
¦
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses

kernel
bias*

Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
8
W	variables
X	keras_api
	Ytotal
	Zcount*
H
[	variables
\	keras_api
	]total
	^count
_
_fn_kwargs*

0
1
2*

0
1
2*
* 


`states
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*
6
ftrace_0
gtrace_1
htrace_2
itrace_3* 
6
jtrace_0
ktrace_1
ltrace_2
mtrace_3* 
* 
ã
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses
t_random_generator
u
state_size

kernel
recurrent_kernel
bias*
* 

0
1
2*

0
1
2*
* 


vstates
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*
6
|trace_0
}trace_1
~trace_2
trace_3* 
:
trace_0
trace_1
trace_2
trace_3* 
* 
ë
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator

state_size

kernel
recurrent_kernel
bias*
* 

0
1
2*

0
1
2*
* 
¥
states
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*
:
trace_0
trace_1
trace_2
trace_3* 
:
trace_0
trace_1
trace_2
trace_3* 
* 
ë
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
 _random_generator
¡
state_size

kernel
recurrent_kernel
bias*
* 

0
1*

0
1*
* 

¢non_trainable_variables
£layers
¤metrics
 ¥layer_regularization_losses
¦layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses*

§trace_0* 

¨trace_0* 
* 
* 
* 

©non_trainable_variables
ªlayers
«metrics
 ¬layer_regularization_losses
­layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses* 

®trace_0* 

¯trace_0* 

Y0
Z1*

W	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

]0
^1*

[	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

70*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1
2*

0
1
2*
* 

°non_trainable_variables
±layers
²metrics
 ³layer_regularization_losses
´layer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses*

µtrace_0
¶trace_1* 

·trace_0
¸trace_1* 
* 
* 
* 
* 

@0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1
2*

0
1
2*
* 

¹non_trainable_variables
ºlayers
»metrics
 ¼layer_regularization_losses
½layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

¾trace_0
¿trace_1* 

Àtrace_0
Átrace_1* 
* 
* 
* 
* 

I0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1
2*

0
1
2*
* 

Ânon_trainable_variables
Ãlayers
Ämetrics
 Ålayer_regularization_losses
Ælayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

Çtrace_0
Ètrace_1* 

Étrace_0
Êtrace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

VARIABLE_VALUE0Adam/rnn_model_21/lstm_45/lstm_cell_183/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE:Adam/rnn_model_21/lstm_45/lstm_cell_183/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE.Adam/rnn_model_21/lstm_45/lstm_cell_183/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE0Adam/rnn_model_21/lstm_46/lstm_cell_184/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE:Adam/rnn_model_21/lstm_46/lstm_cell_184/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE.Adam/rnn_model_21/lstm_46/lstm_cell_184/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE0Adam/rnn_model_21/lstm_47/lstm_cell_185/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE:Adam/rnn_model_21/lstm_47/lstm_cell_185/recurrent_kernel/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE.Adam/rnn_model_21/lstm_47/lstm_cell_185/bias/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/rnn_model_21/dense_21/kernel/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/rnn_model_21/dense_21/bias/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE0Adam/rnn_model_21/lstm_45/lstm_cell_183/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE:Adam/rnn_model_21/lstm_45/lstm_cell_183/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE.Adam/rnn_model_21/lstm_45/lstm_cell_183/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE0Adam/rnn_model_21/lstm_46/lstm_cell_184/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE:Adam/rnn_model_21/lstm_46/lstm_cell_184/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE.Adam/rnn_model_21/lstm_46/lstm_cell_184/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE0Adam/rnn_model_21/lstm_47/lstm_cell_185/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE:Adam/rnn_model_21/lstm_47/lstm_cell_185/recurrent_kernel/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE.Adam/rnn_model_21/lstm_47/lstm_cell_185/bias/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/rnn_model_21/dense_21/kernel/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/rnn_model_21/dense_21/bias/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename=rnn_model_21/lstm_45/lstm_cell_183/kernel/Read/ReadVariableOpGrnn_model_21/lstm_45/lstm_cell_183/recurrent_kernel/Read/ReadVariableOp;rnn_model_21/lstm_45/lstm_cell_183/bias/Read/ReadVariableOp=rnn_model_21/lstm_46/lstm_cell_184/kernel/Read/ReadVariableOpGrnn_model_21/lstm_46/lstm_cell_184/recurrent_kernel/Read/ReadVariableOp;rnn_model_21/lstm_46/lstm_cell_184/bias/Read/ReadVariableOp=rnn_model_21/lstm_47/lstm_cell_185/kernel/Read/ReadVariableOpGrnn_model_21/lstm_47/lstm_cell_185/recurrent_kernel/Read/ReadVariableOp;rnn_model_21/lstm_47/lstm_cell_185/bias/Read/ReadVariableOp0rnn_model_21/dense_21/kernel/Read/ReadVariableOp.rnn_model_21/dense_21/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpDAdam/rnn_model_21/lstm_45/lstm_cell_183/kernel/m/Read/ReadVariableOpNAdam/rnn_model_21/lstm_45/lstm_cell_183/recurrent_kernel/m/Read/ReadVariableOpBAdam/rnn_model_21/lstm_45/lstm_cell_183/bias/m/Read/ReadVariableOpDAdam/rnn_model_21/lstm_46/lstm_cell_184/kernel/m/Read/ReadVariableOpNAdam/rnn_model_21/lstm_46/lstm_cell_184/recurrent_kernel/m/Read/ReadVariableOpBAdam/rnn_model_21/lstm_46/lstm_cell_184/bias/m/Read/ReadVariableOpDAdam/rnn_model_21/lstm_47/lstm_cell_185/kernel/m/Read/ReadVariableOpNAdam/rnn_model_21/lstm_47/lstm_cell_185/recurrent_kernel/m/Read/ReadVariableOpBAdam/rnn_model_21/lstm_47/lstm_cell_185/bias/m/Read/ReadVariableOp7Adam/rnn_model_21/dense_21/kernel/m/Read/ReadVariableOp5Adam/rnn_model_21/dense_21/bias/m/Read/ReadVariableOpDAdam/rnn_model_21/lstm_45/lstm_cell_183/kernel/v/Read/ReadVariableOpNAdam/rnn_model_21/lstm_45/lstm_cell_183/recurrent_kernel/v/Read/ReadVariableOpBAdam/rnn_model_21/lstm_45/lstm_cell_183/bias/v/Read/ReadVariableOpDAdam/rnn_model_21/lstm_46/lstm_cell_184/kernel/v/Read/ReadVariableOpNAdam/rnn_model_21/lstm_46/lstm_cell_184/recurrent_kernel/v/Read/ReadVariableOpBAdam/rnn_model_21/lstm_46/lstm_cell_184/bias/v/Read/ReadVariableOpDAdam/rnn_model_21/lstm_47/lstm_cell_185/kernel/v/Read/ReadVariableOpNAdam/rnn_model_21/lstm_47/lstm_cell_185/recurrent_kernel/v/Read/ReadVariableOpBAdam/rnn_model_21/lstm_47/lstm_cell_185/bias/v/Read/ReadVariableOp7Adam/rnn_model_21/dense_21/kernel/v/Read/ReadVariableOp5Adam/rnn_model_21/dense_21/bias/v/Read/ReadVariableOpConst*7
Tin0
.2,	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_save_3536696
¿
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename)rnn_model_21/lstm_45/lstm_cell_183/kernel3rnn_model_21/lstm_45/lstm_cell_183/recurrent_kernel'rnn_model_21/lstm_45/lstm_cell_183/bias)rnn_model_21/lstm_46/lstm_cell_184/kernel3rnn_model_21/lstm_46/lstm_cell_184/recurrent_kernel'rnn_model_21/lstm_46/lstm_cell_184/bias)rnn_model_21/lstm_47/lstm_cell_185/kernel3rnn_model_21/lstm_47/lstm_cell_185/recurrent_kernel'rnn_model_21/lstm_47/lstm_cell_185/biasrnn_model_21/dense_21/kernelrnn_model_21/dense_21/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcount0Adam/rnn_model_21/lstm_45/lstm_cell_183/kernel/m:Adam/rnn_model_21/lstm_45/lstm_cell_183/recurrent_kernel/m.Adam/rnn_model_21/lstm_45/lstm_cell_183/bias/m0Adam/rnn_model_21/lstm_46/lstm_cell_184/kernel/m:Adam/rnn_model_21/lstm_46/lstm_cell_184/recurrent_kernel/m.Adam/rnn_model_21/lstm_46/lstm_cell_184/bias/m0Adam/rnn_model_21/lstm_47/lstm_cell_185/kernel/m:Adam/rnn_model_21/lstm_47/lstm_cell_185/recurrent_kernel/m.Adam/rnn_model_21/lstm_47/lstm_cell_185/bias/m#Adam/rnn_model_21/dense_21/kernel/m!Adam/rnn_model_21/dense_21/bias/m0Adam/rnn_model_21/lstm_45/lstm_cell_183/kernel/v:Adam/rnn_model_21/lstm_45/lstm_cell_183/recurrent_kernel/v.Adam/rnn_model_21/lstm_45/lstm_cell_183/bias/v0Adam/rnn_model_21/lstm_46/lstm_cell_184/kernel/v:Adam/rnn_model_21/lstm_46/lstm_cell_184/recurrent_kernel/v.Adam/rnn_model_21/lstm_46/lstm_cell_184/bias/v0Adam/rnn_model_21/lstm_47/lstm_cell_185/kernel/v:Adam/rnn_model_21/lstm_47/lstm_cell_185/recurrent_kernel/v.Adam/rnn_model_21/lstm_47/lstm_cell_185/bias/v#Adam/rnn_model_21/dense_21/kernel/v!Adam/rnn_model_21/dense_21/bias/v*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__traced_restore_3536832Æ,
9

D__inference_lstm_47_layer_call_and_return_conditional_losses_3531990

inputs(
lstm_cell_185_3531906:	 (
lstm_cell_185_3531908:	 $
lstm_cell_185_3531910:	
identity¢%lstm_cell_185/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maskþ
%lstm_cell_185/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_185_3531906lstm_cell_185_3531908lstm_cell_185_3531910*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_185_layer_call_and_return_conditional_losses_3531905n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¿
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_185_3531906lstm_cell_185_3531908lstm_cell_185_3531910*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3531920*
condR
while_cond_3531919*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ v
NoOpNoOp&^lstm_cell_185/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : 2N
%lstm_cell_185/StatefulPartitionedCall%lstm_cell_185/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Æ±
« 
#__inference__traced_restore_3536832
file_prefixM
:assignvariableop_rnn_model_21_lstm_45_lstm_cell_183_kernel:	Y
Fassignvariableop_1_rnn_model_21_lstm_45_lstm_cell_183_recurrent_kernel:	 I
:assignvariableop_2_rnn_model_21_lstm_45_lstm_cell_183_bias:	O
<assignvariableop_3_rnn_model_21_lstm_46_lstm_cell_184_kernel:	 Y
Fassignvariableop_4_rnn_model_21_lstm_46_lstm_cell_184_recurrent_kernel:	 I
:assignvariableop_5_rnn_model_21_lstm_46_lstm_cell_184_bias:	O
<assignvariableop_6_rnn_model_21_lstm_47_lstm_cell_185_kernel:	 Y
Fassignvariableop_7_rnn_model_21_lstm_47_lstm_cell_185_recurrent_kernel:	 I
:assignvariableop_8_rnn_model_21_lstm_47_lstm_cell_185_bias:	A
/assignvariableop_9_rnn_model_21_dense_21_kernel: <
.assignvariableop_10_rnn_model_21_dense_21_bias:'
assignvariableop_11_adam_iter:	 )
assignvariableop_12_adam_beta_1: )
assignvariableop_13_adam_beta_2: (
assignvariableop_14_adam_decay: 0
&assignvariableop_15_adam_learning_rate: %
assignvariableop_16_total_1: %
assignvariableop_17_count_1: #
assignvariableop_18_total: #
assignvariableop_19_count: W
Dassignvariableop_20_adam_rnn_model_21_lstm_45_lstm_cell_183_kernel_m:	a
Nassignvariableop_21_adam_rnn_model_21_lstm_45_lstm_cell_183_recurrent_kernel_m:	 Q
Bassignvariableop_22_adam_rnn_model_21_lstm_45_lstm_cell_183_bias_m:	W
Dassignvariableop_23_adam_rnn_model_21_lstm_46_lstm_cell_184_kernel_m:	 a
Nassignvariableop_24_adam_rnn_model_21_lstm_46_lstm_cell_184_recurrent_kernel_m:	 Q
Bassignvariableop_25_adam_rnn_model_21_lstm_46_lstm_cell_184_bias_m:	W
Dassignvariableop_26_adam_rnn_model_21_lstm_47_lstm_cell_185_kernel_m:	 a
Nassignvariableop_27_adam_rnn_model_21_lstm_47_lstm_cell_185_recurrent_kernel_m:	 Q
Bassignvariableop_28_adam_rnn_model_21_lstm_47_lstm_cell_185_bias_m:	I
7assignvariableop_29_adam_rnn_model_21_dense_21_kernel_m: C
5assignvariableop_30_adam_rnn_model_21_dense_21_bias_m:W
Dassignvariableop_31_adam_rnn_model_21_lstm_45_lstm_cell_183_kernel_v:	a
Nassignvariableop_32_adam_rnn_model_21_lstm_45_lstm_cell_183_recurrent_kernel_v:	 Q
Bassignvariableop_33_adam_rnn_model_21_lstm_45_lstm_cell_183_bias_v:	W
Dassignvariableop_34_adam_rnn_model_21_lstm_46_lstm_cell_184_kernel_v:	 a
Nassignvariableop_35_adam_rnn_model_21_lstm_46_lstm_cell_184_recurrent_kernel_v:	 Q
Bassignvariableop_36_adam_rnn_model_21_lstm_46_lstm_cell_184_bias_v:	W
Dassignvariableop_37_adam_rnn_model_21_lstm_47_lstm_cell_185_kernel_v:	 a
Nassignvariableop_38_adam_rnn_model_21_lstm_47_lstm_cell_185_recurrent_kernel_v:	 Q
Bassignvariableop_39_adam_rnn_model_21_lstm_47_lstm_cell_185_bias_v:	I
7assignvariableop_40_adam_rnn_model_21_dense_21_kernel_v: C
5assignvariableop_41_adam_rnn_model_21_dense_21_bias_v:
identity_43¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Ý
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*
valueùBö+B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÆ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ø
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Â
_output_shapes¯
¬:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOpAssignVariableOp:assignvariableop_rnn_model_21_lstm_45_lstm_cell_183_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:µ
AssignVariableOp_1AssignVariableOpFassignvariableop_1_rnn_model_21_lstm_45_lstm_cell_183_recurrent_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_2AssignVariableOp:assignvariableop_2_rnn_model_21_lstm_45_lstm_cell_183_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_3AssignVariableOp<assignvariableop_3_rnn_model_21_lstm_46_lstm_cell_184_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:µ
AssignVariableOp_4AssignVariableOpFassignvariableop_4_rnn_model_21_lstm_46_lstm_cell_184_recurrent_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_5AssignVariableOp:assignvariableop_5_rnn_model_21_lstm_46_lstm_cell_184_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_6AssignVariableOp<assignvariableop_6_rnn_model_21_lstm_47_lstm_cell_185_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:µ
AssignVariableOp_7AssignVariableOpFassignvariableop_7_rnn_model_21_lstm_47_lstm_cell_185_recurrent_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_8AssignVariableOp:assignvariableop_8_rnn_model_21_lstm_47_lstm_cell_185_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp/assignvariableop_9_rnn_model_21_dense_21_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp.assignvariableop_10_rnn_model_21_dense_21_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_iterIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_2Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_decayIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp&assignvariableop_15_adam_learning_rateIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_total_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_count_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_totalIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOpassignvariableop_19_countIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:µ
AssignVariableOp_20AssignVariableOpDassignvariableop_20_adam_rnn_model_21_lstm_45_lstm_cell_183_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:¿
AssignVariableOp_21AssignVariableOpNassignvariableop_21_adam_rnn_model_21_lstm_45_lstm_cell_183_recurrent_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:³
AssignVariableOp_22AssignVariableOpBassignvariableop_22_adam_rnn_model_21_lstm_45_lstm_cell_183_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:µ
AssignVariableOp_23AssignVariableOpDassignvariableop_23_adam_rnn_model_21_lstm_46_lstm_cell_184_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:¿
AssignVariableOp_24AssignVariableOpNassignvariableop_24_adam_rnn_model_21_lstm_46_lstm_cell_184_recurrent_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:³
AssignVariableOp_25AssignVariableOpBassignvariableop_25_adam_rnn_model_21_lstm_46_lstm_cell_184_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:µ
AssignVariableOp_26AssignVariableOpDassignvariableop_26_adam_rnn_model_21_lstm_47_lstm_cell_185_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:¿
AssignVariableOp_27AssignVariableOpNassignvariableop_27_adam_rnn_model_21_lstm_47_lstm_cell_185_recurrent_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:³
AssignVariableOp_28AssignVariableOpBassignvariableop_28_adam_rnn_model_21_lstm_47_lstm_cell_185_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_29AssignVariableOp7assignvariableop_29_adam_rnn_model_21_dense_21_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_30AssignVariableOp5assignvariableop_30_adam_rnn_model_21_dense_21_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:µ
AssignVariableOp_31AssignVariableOpDassignvariableop_31_adam_rnn_model_21_lstm_45_lstm_cell_183_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:¿
AssignVariableOp_32AssignVariableOpNassignvariableop_32_adam_rnn_model_21_lstm_45_lstm_cell_183_recurrent_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:³
AssignVariableOp_33AssignVariableOpBassignvariableop_33_adam_rnn_model_21_lstm_45_lstm_cell_183_bias_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:µ
AssignVariableOp_34AssignVariableOpDassignvariableop_34_adam_rnn_model_21_lstm_46_lstm_cell_184_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:¿
AssignVariableOp_35AssignVariableOpNassignvariableop_35_adam_rnn_model_21_lstm_46_lstm_cell_184_recurrent_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:³
AssignVariableOp_36AssignVariableOpBassignvariableop_36_adam_rnn_model_21_lstm_46_lstm_cell_184_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:µ
AssignVariableOp_37AssignVariableOpDassignvariableop_37_adam_rnn_model_21_lstm_47_lstm_cell_185_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:¿
AssignVariableOp_38AssignVariableOpNassignvariableop_38_adam_rnn_model_21_lstm_47_lstm_cell_185_recurrent_kernel_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:³
AssignVariableOp_39AssignVariableOpBassignvariableop_39_adam_rnn_model_21_lstm_47_lstm_cell_185_bias_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_40AssignVariableOp7assignvariableop_40_adam_rnn_model_21_dense_21_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_41AssignVariableOp5assignvariableop_41_adam_rnn_model_21_dense_21_bias_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ë
Identity_42Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_43IdentityIdentity_42:output:0^NoOp_1*
T0*
_output_shapes
: Ø
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_43Identity_43:output:0*i
_input_shapesX
V: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
«
¸
)__inference_lstm_45_layer_call_fn_3534382
inputs_0
unknown:	
	unknown_0:	 
	unknown_1:	
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_45_layer_call_and_return_conditional_losses_3531479|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
®8

D__inference_lstm_45_layer_call_and_return_conditional_losses_3531288

inputs(
lstm_cell_183_3531206:	(
lstm_cell_183_3531208:	 $
lstm_cell_183_3531210:	
identity¢%lstm_cell_183/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskþ
%lstm_cell_183/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_183_3531206lstm_cell_183_3531208lstm_cell_183_3531210*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_183_layer_call_and_return_conditional_losses_3531205n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¿
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_183_3531206lstm_cell_183_3531208lstm_cell_183_3531210*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3531219*
condR
while_cond_3531218*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ v
NoOpNoOp&^lstm_cell_183/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2N
%lstm_cell_183/StatefulPartitionedCall%lstm_cell_183/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
åR
û
'rnn_model_21_lstm_47_while_body_3531038F
Brnn_model_21_lstm_47_while_rnn_model_21_lstm_47_while_loop_counterL
Hrnn_model_21_lstm_47_while_rnn_model_21_lstm_47_while_maximum_iterations*
&rnn_model_21_lstm_47_while_placeholder,
(rnn_model_21_lstm_47_while_placeholder_1,
(rnn_model_21_lstm_47_while_placeholder_2,
(rnn_model_21_lstm_47_while_placeholder_3E
Arnn_model_21_lstm_47_while_rnn_model_21_lstm_47_strided_slice_1_0
}rnn_model_21_lstm_47_while_tensorarrayv2read_tensorlistgetitem_rnn_model_21_lstm_47_tensorarrayunstack_tensorlistfromtensor_0\
Irnn_model_21_lstm_47_while_lstm_cell_185_matmul_readvariableop_resource_0:	 ^
Krnn_model_21_lstm_47_while_lstm_cell_185_matmul_1_readvariableop_resource_0:	 Y
Jrnn_model_21_lstm_47_while_lstm_cell_185_biasadd_readvariableop_resource_0:	'
#rnn_model_21_lstm_47_while_identity)
%rnn_model_21_lstm_47_while_identity_1)
%rnn_model_21_lstm_47_while_identity_2)
%rnn_model_21_lstm_47_while_identity_3)
%rnn_model_21_lstm_47_while_identity_4)
%rnn_model_21_lstm_47_while_identity_5C
?rnn_model_21_lstm_47_while_rnn_model_21_lstm_47_strided_slice_1
{rnn_model_21_lstm_47_while_tensorarrayv2read_tensorlistgetitem_rnn_model_21_lstm_47_tensorarrayunstack_tensorlistfromtensorZ
Grnn_model_21_lstm_47_while_lstm_cell_185_matmul_readvariableop_resource:	 \
Irnn_model_21_lstm_47_while_lstm_cell_185_matmul_1_readvariableop_resource:	 W
Hrnn_model_21_lstm_47_while_lstm_cell_185_biasadd_readvariableop_resource:	¢?rnn_model_21/lstm_47/while/lstm_cell_185/BiasAdd/ReadVariableOp¢>rnn_model_21/lstm_47/while/lstm_cell_185/MatMul/ReadVariableOp¢@rnn_model_21/lstm_47/while/lstm_cell_185/MatMul_1/ReadVariableOp
Lrnn_model_21/lstm_47/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    
>rnn_model_21/lstm_47/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}rnn_model_21_lstm_47_while_tensorarrayv2read_tensorlistgetitem_rnn_model_21_lstm_47_tensorarrayunstack_tensorlistfromtensor_0&rnn_model_21_lstm_47_while_placeholderUrnn_model_21/lstm_47/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0É
>rnn_model_21/lstm_47/while/lstm_cell_185/MatMul/ReadVariableOpReadVariableOpIrnn_model_21_lstm_47_while_lstm_cell_185_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype0û
/rnn_model_21/lstm_47/while/lstm_cell_185/MatMulMatMulErnn_model_21/lstm_47/while/TensorArrayV2Read/TensorListGetItem:item:0Frnn_model_21/lstm_47/while/lstm_cell_185/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ
@rnn_model_21/lstm_47/while/lstm_cell_185/MatMul_1/ReadVariableOpReadVariableOpKrnn_model_21_lstm_47_while_lstm_cell_185_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0â
1rnn_model_21/lstm_47/while/lstm_cell_185/MatMul_1MatMul(rnn_model_21_lstm_47_while_placeholder_2Hrnn_model_21/lstm_47/while/lstm_cell_185/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
,rnn_model_21/lstm_47/while/lstm_cell_185/addAddV29rnn_model_21/lstm_47/while/lstm_cell_185/MatMul:product:0;rnn_model_21/lstm_47/while/lstm_cell_185/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ
?rnn_model_21/lstm_47/while/lstm_cell_185/BiasAdd/ReadVariableOpReadVariableOpJrnn_model_21_lstm_47_while_lstm_cell_185_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0é
0rnn_model_21/lstm_47/while/lstm_cell_185/BiasAddBiasAdd0rnn_model_21/lstm_47/while/lstm_cell_185/add:z:0Grnn_model_21/lstm_47/while/lstm_cell_185/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
8rnn_model_21/lstm_47/while/lstm_cell_185/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :±
.rnn_model_21/lstm_47/while/lstm_cell_185/splitSplitArnn_model_21/lstm_47/while/lstm_cell_185/split/split_dim:output:09rnn_model_21/lstm_47/while/lstm_cell_185/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split¦
0rnn_model_21/lstm_47/while/lstm_cell_185/SigmoidSigmoid7rnn_model_21/lstm_47/while/lstm_cell_185/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¨
2rnn_model_21/lstm_47/while/lstm_cell_185/Sigmoid_1Sigmoid7rnn_model_21/lstm_47/while/lstm_cell_185/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ç
,rnn_model_21/lstm_47/while/lstm_cell_185/mulMul6rnn_model_21/lstm_47/while/lstm_cell_185/Sigmoid_1:y:0(rnn_model_21_lstm_47_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
-rnn_model_21/lstm_47/while/lstm_cell_185/ReluRelu7rnn_model_21/lstm_47/while/lstm_cell_185/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ú
.rnn_model_21/lstm_47/while/lstm_cell_185/mul_1Mul4rnn_model_21/lstm_47/while/lstm_cell_185/Sigmoid:y:0;rnn_model_21/lstm_47/while/lstm_cell_185/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ï
.rnn_model_21/lstm_47/while/lstm_cell_185/add_1AddV20rnn_model_21/lstm_47/while/lstm_cell_185/mul:z:02rnn_model_21/lstm_47/while/lstm_cell_185/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¨
2rnn_model_21/lstm_47/while/lstm_cell_185/Sigmoid_2Sigmoid7rnn_model_21/lstm_47/while/lstm_cell_185/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
/rnn_model_21/lstm_47/while/lstm_cell_185/Relu_1Relu2rnn_model_21/lstm_47/while/lstm_cell_185/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Þ
.rnn_model_21/lstm_47/while/lstm_cell_185/mul_2Mul6rnn_model_21/lstm_47/while/lstm_cell_185/Sigmoid_2:y:0=rnn_model_21/lstm_47/while/lstm_cell_185/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
Ernn_model_21/lstm_47/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Â
?rnn_model_21/lstm_47/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(rnn_model_21_lstm_47_while_placeholder_1Nrnn_model_21/lstm_47/while/TensorArrayV2Write/TensorListSetItem/index:output:02rnn_model_21/lstm_47/while/lstm_cell_185/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒb
 rnn_model_21/lstm_47/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
rnn_model_21/lstm_47/while/addAddV2&rnn_model_21_lstm_47_while_placeholder)rnn_model_21/lstm_47/while/add/y:output:0*
T0*
_output_shapes
: d
"rnn_model_21/lstm_47/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :»
 rnn_model_21/lstm_47/while/add_1AddV2Brnn_model_21_lstm_47_while_rnn_model_21_lstm_47_while_loop_counter+rnn_model_21/lstm_47/while/add_1/y:output:0*
T0*
_output_shapes
: 
#rnn_model_21/lstm_47/while/IdentityIdentity$rnn_model_21/lstm_47/while/add_1:z:0 ^rnn_model_21/lstm_47/while/NoOp*
T0*
_output_shapes
: ¾
%rnn_model_21/lstm_47/while/Identity_1IdentityHrnn_model_21_lstm_47_while_rnn_model_21_lstm_47_while_maximum_iterations ^rnn_model_21/lstm_47/while/NoOp*
T0*
_output_shapes
: 
%rnn_model_21/lstm_47/while/Identity_2Identity"rnn_model_21/lstm_47/while/add:z:0 ^rnn_model_21/lstm_47/while/NoOp*
T0*
_output_shapes
: Å
%rnn_model_21/lstm_47/while/Identity_3IdentityOrnn_model_21/lstm_47/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^rnn_model_21/lstm_47/while/NoOp*
T0*
_output_shapes
: ¹
%rnn_model_21/lstm_47/while/Identity_4Identity2rnn_model_21/lstm_47/while/lstm_cell_185/mul_2:z:0 ^rnn_model_21/lstm_47/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¹
%rnn_model_21/lstm_47/while/Identity_5Identity2rnn_model_21/lstm_47/while/lstm_cell_185/add_1:z:0 ^rnn_model_21/lstm_47/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ §
rnn_model_21/lstm_47/while/NoOpNoOp@^rnn_model_21/lstm_47/while/lstm_cell_185/BiasAdd/ReadVariableOp?^rnn_model_21/lstm_47/while/lstm_cell_185/MatMul/ReadVariableOpA^rnn_model_21/lstm_47/while/lstm_cell_185/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "S
#rnn_model_21_lstm_47_while_identity,rnn_model_21/lstm_47/while/Identity:output:0"W
%rnn_model_21_lstm_47_while_identity_1.rnn_model_21/lstm_47/while/Identity_1:output:0"W
%rnn_model_21_lstm_47_while_identity_2.rnn_model_21/lstm_47/while/Identity_2:output:0"W
%rnn_model_21_lstm_47_while_identity_3.rnn_model_21/lstm_47/while/Identity_3:output:0"W
%rnn_model_21_lstm_47_while_identity_4.rnn_model_21/lstm_47/while/Identity_4:output:0"W
%rnn_model_21_lstm_47_while_identity_5.rnn_model_21/lstm_47/while/Identity_5:output:0"
Hrnn_model_21_lstm_47_while_lstm_cell_185_biasadd_readvariableop_resourceJrnn_model_21_lstm_47_while_lstm_cell_185_biasadd_readvariableop_resource_0"
Irnn_model_21_lstm_47_while_lstm_cell_185_matmul_1_readvariableop_resourceKrnn_model_21_lstm_47_while_lstm_cell_185_matmul_1_readvariableop_resource_0"
Grnn_model_21_lstm_47_while_lstm_cell_185_matmul_readvariableop_resourceIrnn_model_21_lstm_47_while_lstm_cell_185_matmul_readvariableop_resource_0"
?rnn_model_21_lstm_47_while_rnn_model_21_lstm_47_strided_slice_1Arnn_model_21_lstm_47_while_rnn_model_21_lstm_47_strided_slice_1_0"ü
{rnn_model_21_lstm_47_while_tensorarrayv2read_tensorlistgetitem_rnn_model_21_lstm_47_tensorarrayunstack_tensorlistfromtensor}rnn_model_21_lstm_47_while_tensorarrayv2read_tensorlistgetitem_rnn_model_21_lstm_47_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2
?rnn_model_21/lstm_47/while/lstm_cell_185/BiasAdd/ReadVariableOp?rnn_model_21/lstm_47/while/lstm_cell_185/BiasAdd/ReadVariableOp2
>rnn_model_21/lstm_47/while/lstm_cell_185/MatMul/ReadVariableOp>rnn_model_21/lstm_47/while/lstm_cell_185/MatMul/ReadVariableOp2
@rnn_model_21/lstm_47/while/lstm_cell_185/MatMul_1/ReadVariableOp@rnn_model_21/lstm_47/while/lstm_cell_185/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
áJ
¢
D__inference_lstm_45_layer_call_and_return_conditional_losses_3533212

inputs?
,lstm_cell_183_matmul_readvariableop_resource:	A
.lstm_cell_183_matmul_1_readvariableop_resource:	 <
-lstm_cell_183_biasadd_readvariableop_resource:	
identity¢$lstm_cell_183/BiasAdd/ReadVariableOp¢#lstm_cell_183/MatMul/ReadVariableOp¢%lstm_cell_183/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
#lstm_cell_183/MatMul/ReadVariableOpReadVariableOp,lstm_cell_183_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_183/MatMulMatMulstrided_slice_2:output:0+lstm_cell_183/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%lstm_cell_183/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_183_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_183/MatMul_1MatMulzeros:output:0-lstm_cell_183/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_183/addAddV2lstm_cell_183/MatMul:product:0 lstm_cell_183/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_183/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_183_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_183/BiasAddBiasAddlstm_cell_183/add:z:0,lstm_cell_183/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
lstm_cell_183/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :à
lstm_cell_183/splitSplit&lstm_cell_183/split/split_dim:output:0lstm_cell_183/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitp
lstm_cell_183/SigmoidSigmoidlstm_cell_183/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
lstm_cell_183/Sigmoid_1Sigmoidlstm_cell_183/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y
lstm_cell_183/mulMullstm_cell_183/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
lstm_cell_183/ReluRelulstm_cell_183/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_183/mul_1Mullstm_cell_183/Sigmoid:y:0 lstm_cell_183/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
lstm_cell_183/add_1AddV2lstm_cell_183/mul:z:0lstm_cell_183/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
lstm_cell_183/Sigmoid_2Sigmoidlstm_cell_183/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ g
lstm_cell_183/Relu_1Relulstm_cell_183/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_183/mul_2Mullstm_cell_183/Sigmoid_2:y:0"lstm_cell_183/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_183_matmul_readvariableop_resource.lstm_cell_183_matmul_1_readvariableop_resource-lstm_cell_183_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3533128*
condR
while_cond_3533127*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 Ã
NoOpNoOp%^lstm_cell_183/BiasAdd/ReadVariableOp$^lstm_cell_183/MatMul/ReadVariableOp&^lstm_cell_183/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : : 2L
$lstm_cell_183/BiasAdd/ReadVariableOp$lstm_cell_183/BiasAdd/ReadVariableOp2J
#lstm_cell_183/MatMul/ReadVariableOp#lstm_cell_183/MatMul/ReadVariableOp2N
%lstm_cell_183/MatMul_1/ReadVariableOp%lstm_cell_183/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
ß

J__inference_lstm_cell_183_layer_call_and_return_conditional_losses_3536351

inputs
states_0
states_11
matmul_readvariableop_resource:	3
 matmul_1_readvariableop_resource:	 .
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ N
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/1
Ä

*__inference_dense_21_layer_call_fn_3536225

inputs
unknown: 
	unknown_0:
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_3532661o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
B
Ú

lstm_45_while_body_3533543,
(lstm_45_while_lstm_45_while_loop_counter2
.lstm_45_while_lstm_45_while_maximum_iterations
lstm_45_while_placeholder
lstm_45_while_placeholder_1
lstm_45_while_placeholder_2
lstm_45_while_placeholder_3+
'lstm_45_while_lstm_45_strided_slice_1_0g
clstm_45_while_tensorarrayv2read_tensorlistgetitem_lstm_45_tensorarrayunstack_tensorlistfromtensor_0O
<lstm_45_while_lstm_cell_183_matmul_readvariableop_resource_0:	Q
>lstm_45_while_lstm_cell_183_matmul_1_readvariableop_resource_0:	 L
=lstm_45_while_lstm_cell_183_biasadd_readvariableop_resource_0:	
lstm_45_while_identity
lstm_45_while_identity_1
lstm_45_while_identity_2
lstm_45_while_identity_3
lstm_45_while_identity_4
lstm_45_while_identity_5)
%lstm_45_while_lstm_45_strided_slice_1e
alstm_45_while_tensorarrayv2read_tensorlistgetitem_lstm_45_tensorarrayunstack_tensorlistfromtensorM
:lstm_45_while_lstm_cell_183_matmul_readvariableop_resource:	O
<lstm_45_while_lstm_cell_183_matmul_1_readvariableop_resource:	 J
;lstm_45_while_lstm_cell_183_biasadd_readvariableop_resource:	¢2lstm_45/while/lstm_cell_183/BiasAdd/ReadVariableOp¢1lstm_45/while/lstm_cell_183/MatMul/ReadVariableOp¢3lstm_45/while/lstm_cell_183/MatMul_1/ReadVariableOp
?lstm_45/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Î
1lstm_45/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_45_while_tensorarrayv2read_tensorlistgetitem_lstm_45_tensorarrayunstack_tensorlistfromtensor_0lstm_45_while_placeholderHlstm_45/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¯
1lstm_45/while/lstm_cell_183/MatMul/ReadVariableOpReadVariableOp<lstm_45_while_lstm_cell_183_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0Ô
"lstm_45/while/lstm_cell_183/MatMulMatMul8lstm_45/while/TensorArrayV2Read/TensorListGetItem:item:09lstm_45/while/lstm_cell_183/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
3lstm_45/while/lstm_cell_183/MatMul_1/ReadVariableOpReadVariableOp>lstm_45_while_lstm_cell_183_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0»
$lstm_45/while/lstm_cell_183/MatMul_1MatMullstm_45_while_placeholder_2;lstm_45/while/lstm_cell_183/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
lstm_45/while/lstm_cell_183/addAddV2,lstm_45/while/lstm_cell_183/MatMul:product:0.lstm_45/while/lstm_cell_183/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
2lstm_45/while/lstm_cell_183/BiasAdd/ReadVariableOpReadVariableOp=lstm_45_while_lstm_cell_183_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0Â
#lstm_45/while/lstm_cell_183/BiasAddBiasAdd#lstm_45/while/lstm_cell_183/add:z:0:lstm_45/while/lstm_cell_183/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
+lstm_45/while/lstm_cell_183/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
!lstm_45/while/lstm_cell_183/splitSplit4lstm_45/while/lstm_cell_183/split/split_dim:output:0,lstm_45/while/lstm_cell_183/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split
#lstm_45/while/lstm_cell_183/SigmoidSigmoid*lstm_45/while/lstm_cell_183/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
%lstm_45/while/lstm_cell_183/Sigmoid_1Sigmoid*lstm_45/while/lstm_cell_183/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
lstm_45/while/lstm_cell_183/mulMul)lstm_45/while/lstm_cell_183/Sigmoid_1:y:0lstm_45_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 lstm_45/while/lstm_cell_183/ReluRelu*lstm_45/while/lstm_cell_183/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ³
!lstm_45/while/lstm_cell_183/mul_1Mul'lstm_45/while/lstm_cell_183/Sigmoid:y:0.lstm_45/while/lstm_cell_183/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¨
!lstm_45/while/lstm_cell_183/add_1AddV2#lstm_45/while/lstm_cell_183/mul:z:0%lstm_45/while/lstm_cell_183/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
%lstm_45/while/lstm_cell_183/Sigmoid_2Sigmoid*lstm_45/while/lstm_cell_183/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"lstm_45/while/lstm_cell_183/Relu_1Relu%lstm_45/while/lstm_cell_183/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ·
!lstm_45/while/lstm_cell_183/mul_2Mul)lstm_45/while/lstm_cell_183/Sigmoid_2:y:00lstm_45/while/lstm_cell_183/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ æ
2lstm_45/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_45_while_placeholder_1lstm_45_while_placeholder%lstm_45/while/lstm_cell_183/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒU
lstm_45/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_45/while/addAddV2lstm_45_while_placeholderlstm_45/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_45/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_45/while/add_1AddV2(lstm_45_while_lstm_45_while_loop_counterlstm_45/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_45/while/IdentityIdentitylstm_45/while/add_1:z:0^lstm_45/while/NoOp*
T0*
_output_shapes
: 
lstm_45/while/Identity_1Identity.lstm_45_while_lstm_45_while_maximum_iterations^lstm_45/while/NoOp*
T0*
_output_shapes
: q
lstm_45/while/Identity_2Identitylstm_45/while/add:z:0^lstm_45/while/NoOp*
T0*
_output_shapes
: 
lstm_45/while/Identity_3IdentityBlstm_45/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_45/while/NoOp*
T0*
_output_shapes
: 
lstm_45/while/Identity_4Identity%lstm_45/while/lstm_cell_183/mul_2:z:0^lstm_45/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_45/while/Identity_5Identity%lstm_45/while/lstm_cell_183/add_1:z:0^lstm_45/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ó
lstm_45/while/NoOpNoOp3^lstm_45/while/lstm_cell_183/BiasAdd/ReadVariableOp2^lstm_45/while/lstm_cell_183/MatMul/ReadVariableOp4^lstm_45/while/lstm_cell_183/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_45_while_identitylstm_45/while/Identity:output:0"=
lstm_45_while_identity_1!lstm_45/while/Identity_1:output:0"=
lstm_45_while_identity_2!lstm_45/while/Identity_2:output:0"=
lstm_45_while_identity_3!lstm_45/while/Identity_3:output:0"=
lstm_45_while_identity_4!lstm_45/while/Identity_4:output:0"=
lstm_45_while_identity_5!lstm_45/while/Identity_5:output:0"P
%lstm_45_while_lstm_45_strided_slice_1'lstm_45_while_lstm_45_strided_slice_1_0"|
;lstm_45_while_lstm_cell_183_biasadd_readvariableop_resource=lstm_45_while_lstm_cell_183_biasadd_readvariableop_resource_0"~
<lstm_45_while_lstm_cell_183_matmul_1_readvariableop_resource>lstm_45_while_lstm_cell_183_matmul_1_readvariableop_resource_0"z
:lstm_45_while_lstm_cell_183_matmul_readvariableop_resource<lstm_45_while_lstm_cell_183_matmul_readvariableop_resource_0"È
alstm_45_while_tensorarrayv2read_tensorlistgetitem_lstm_45_tensorarrayunstack_tensorlistfromtensorclstm_45_while_tensorarrayv2read_tensorlistgetitem_lstm_45_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2h
2lstm_45/while/lstm_cell_183/BiasAdd/ReadVariableOp2lstm_45/while/lstm_cell_183/BiasAdd/ReadVariableOp2f
1lstm_45/while/lstm_cell_183/MatMul/ReadVariableOp1lstm_45/while/lstm_cell_183/MatMul/ReadVariableOp2j
3lstm_45/while/lstm_cell_183/MatMul_1/ReadVariableOp3lstm_45/while/lstm_cell_183/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 

¸
)__inference_lstm_47_layer_call_fn_3535603
inputs_0
unknown:	 
	unknown_0:	 
	unknown_1:	
identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_47_layer_call_and_return_conditional_losses_3531990o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0
º
È
while_cond_3535364
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3535364___redundant_placeholder05
1while_while_cond_3535364___redundant_placeholder15
1while_while_cond_3535364___redundant_placeholder25
1while_while_cond_3535364___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
K
¤
D__inference_lstm_45_layer_call_and_return_conditional_losses_3534690
inputs_0?
,lstm_cell_183_matmul_readvariableop_resource:	A
.lstm_cell_183_matmul_1_readvariableop_resource:	 <
-lstm_cell_183_biasadd_readvariableop_resource:	
identity¢$lstm_cell_183/BiasAdd/ReadVariableOp¢#lstm_cell_183/MatMul/ReadVariableOp¢%lstm_cell_183/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
#lstm_cell_183/MatMul/ReadVariableOpReadVariableOp,lstm_cell_183_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_183/MatMulMatMulstrided_slice_2:output:0+lstm_cell_183/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%lstm_cell_183/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_183_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_183/MatMul_1MatMulzeros:output:0-lstm_cell_183/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_183/addAddV2lstm_cell_183/MatMul:product:0 lstm_cell_183/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_183/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_183_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_183/BiasAddBiasAddlstm_cell_183/add:z:0,lstm_cell_183/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
lstm_cell_183/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :à
lstm_cell_183/splitSplit&lstm_cell_183/split/split_dim:output:0lstm_cell_183/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitp
lstm_cell_183/SigmoidSigmoidlstm_cell_183/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
lstm_cell_183/Sigmoid_1Sigmoidlstm_cell_183/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y
lstm_cell_183/mulMullstm_cell_183/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
lstm_cell_183/ReluRelulstm_cell_183/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_183/mul_1Mullstm_cell_183/Sigmoid:y:0 lstm_cell_183/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
lstm_cell_183/add_1AddV2lstm_cell_183/mul:z:0lstm_cell_183/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
lstm_cell_183/Sigmoid_2Sigmoidlstm_cell_183/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ g
lstm_cell_183/Relu_1Relulstm_cell_183/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_183/mul_2Mullstm_cell_183/Sigmoid_2:y:0"lstm_cell_183/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_183_matmul_readvariableop_resource.lstm_cell_183_matmul_1_readvariableop_resource-lstm_cell_183_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3534606*
condR
while_cond_3534605*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ã
NoOpNoOp%^lstm_cell_183/BiasAdd/ReadVariableOp$^lstm_cell_183/MatMul/ReadVariableOp&^lstm_cell_183/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_183/BiasAdd/ReadVariableOp$lstm_cell_183/BiasAdd/ReadVariableOp2J
#lstm_cell_183/MatMul/ReadVariableOp#lstm_cell_183/MatMul/ReadVariableOp2N
%lstm_cell_183/MatMul_1/ReadVariableOp%lstm_cell_183/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
º
È
while_cond_3535695
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3535695___redundant_placeholder05
1while_while_cond_3535695___redundant_placeholder15
1while_while_cond_3535695___redundant_placeholder25
1while_while_cond_3535695___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:

¶
)__inference_lstm_45_layer_call_fn_3534404

inputs
unknown:	
	unknown_0:	 
	unknown_1:	
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_45_layer_call_and_return_conditional_losses_3533212s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
º
È
while_cond_3532796
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3532796___redundant_placeholder05
1while_while_cond_3532796___redundant_placeholder15
1while_while_cond_3532796___redundant_placeholder25
1while_while_cond_3532796___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
B
Ú

lstm_46_while_body_3533682,
(lstm_46_while_lstm_46_while_loop_counter2
.lstm_46_while_lstm_46_while_maximum_iterations
lstm_46_while_placeholder
lstm_46_while_placeholder_1
lstm_46_while_placeholder_2
lstm_46_while_placeholder_3+
'lstm_46_while_lstm_46_strided_slice_1_0g
clstm_46_while_tensorarrayv2read_tensorlistgetitem_lstm_46_tensorarrayunstack_tensorlistfromtensor_0O
<lstm_46_while_lstm_cell_184_matmul_readvariableop_resource_0:	 Q
>lstm_46_while_lstm_cell_184_matmul_1_readvariableop_resource_0:	 L
=lstm_46_while_lstm_cell_184_biasadd_readvariableop_resource_0:	
lstm_46_while_identity
lstm_46_while_identity_1
lstm_46_while_identity_2
lstm_46_while_identity_3
lstm_46_while_identity_4
lstm_46_while_identity_5)
%lstm_46_while_lstm_46_strided_slice_1e
alstm_46_while_tensorarrayv2read_tensorlistgetitem_lstm_46_tensorarrayunstack_tensorlistfromtensorM
:lstm_46_while_lstm_cell_184_matmul_readvariableop_resource:	 O
<lstm_46_while_lstm_cell_184_matmul_1_readvariableop_resource:	 J
;lstm_46_while_lstm_cell_184_biasadd_readvariableop_resource:	¢2lstm_46/while/lstm_cell_184/BiasAdd/ReadVariableOp¢1lstm_46/while/lstm_cell_184/MatMul/ReadVariableOp¢3lstm_46/while/lstm_cell_184/MatMul_1/ReadVariableOp
?lstm_46/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Î
1lstm_46/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_46_while_tensorarrayv2read_tensorlistgetitem_lstm_46_tensorarrayunstack_tensorlistfromtensor_0lstm_46_while_placeholderHlstm_46/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0¯
1lstm_46/while/lstm_cell_184/MatMul/ReadVariableOpReadVariableOp<lstm_46_while_lstm_cell_184_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype0Ô
"lstm_46/while/lstm_cell_184/MatMulMatMul8lstm_46/while/TensorArrayV2Read/TensorListGetItem:item:09lstm_46/while/lstm_cell_184/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
3lstm_46/while/lstm_cell_184/MatMul_1/ReadVariableOpReadVariableOp>lstm_46_while_lstm_cell_184_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0»
$lstm_46/while/lstm_cell_184/MatMul_1MatMullstm_46_while_placeholder_2;lstm_46/while/lstm_cell_184/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
lstm_46/while/lstm_cell_184/addAddV2,lstm_46/while/lstm_cell_184/MatMul:product:0.lstm_46/while/lstm_cell_184/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
2lstm_46/while/lstm_cell_184/BiasAdd/ReadVariableOpReadVariableOp=lstm_46_while_lstm_cell_184_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0Â
#lstm_46/while/lstm_cell_184/BiasAddBiasAdd#lstm_46/while/lstm_cell_184/add:z:0:lstm_46/while/lstm_cell_184/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
+lstm_46/while/lstm_cell_184/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
!lstm_46/while/lstm_cell_184/splitSplit4lstm_46/while/lstm_cell_184/split/split_dim:output:0,lstm_46/while/lstm_cell_184/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split
#lstm_46/while/lstm_cell_184/SigmoidSigmoid*lstm_46/while/lstm_cell_184/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
%lstm_46/while/lstm_cell_184/Sigmoid_1Sigmoid*lstm_46/while/lstm_cell_184/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
lstm_46/while/lstm_cell_184/mulMul)lstm_46/while/lstm_cell_184/Sigmoid_1:y:0lstm_46_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 lstm_46/while/lstm_cell_184/ReluRelu*lstm_46/while/lstm_cell_184/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ³
!lstm_46/while/lstm_cell_184/mul_1Mul'lstm_46/while/lstm_cell_184/Sigmoid:y:0.lstm_46/while/lstm_cell_184/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¨
!lstm_46/while/lstm_cell_184/add_1AddV2#lstm_46/while/lstm_cell_184/mul:z:0%lstm_46/while/lstm_cell_184/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
%lstm_46/while/lstm_cell_184/Sigmoid_2Sigmoid*lstm_46/while/lstm_cell_184/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"lstm_46/while/lstm_cell_184/Relu_1Relu%lstm_46/while/lstm_cell_184/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ·
!lstm_46/while/lstm_cell_184/mul_2Mul)lstm_46/while/lstm_cell_184/Sigmoid_2:y:00lstm_46/while/lstm_cell_184/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ æ
2lstm_46/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_46_while_placeholder_1lstm_46_while_placeholder%lstm_46/while/lstm_cell_184/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒU
lstm_46/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_46/while/addAddV2lstm_46_while_placeholderlstm_46/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_46/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_46/while/add_1AddV2(lstm_46_while_lstm_46_while_loop_counterlstm_46/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_46/while/IdentityIdentitylstm_46/while/add_1:z:0^lstm_46/while/NoOp*
T0*
_output_shapes
: 
lstm_46/while/Identity_1Identity.lstm_46_while_lstm_46_while_maximum_iterations^lstm_46/while/NoOp*
T0*
_output_shapes
: q
lstm_46/while/Identity_2Identitylstm_46/while/add:z:0^lstm_46/while/NoOp*
T0*
_output_shapes
: 
lstm_46/while/Identity_3IdentityBlstm_46/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_46/while/NoOp*
T0*
_output_shapes
: 
lstm_46/while/Identity_4Identity%lstm_46/while/lstm_cell_184/mul_2:z:0^lstm_46/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_46/while/Identity_5Identity%lstm_46/while/lstm_cell_184/add_1:z:0^lstm_46/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ó
lstm_46/while/NoOpNoOp3^lstm_46/while/lstm_cell_184/BiasAdd/ReadVariableOp2^lstm_46/while/lstm_cell_184/MatMul/ReadVariableOp4^lstm_46/while/lstm_cell_184/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_46_while_identitylstm_46/while/Identity:output:0"=
lstm_46_while_identity_1!lstm_46/while/Identity_1:output:0"=
lstm_46_while_identity_2!lstm_46/while/Identity_2:output:0"=
lstm_46_while_identity_3!lstm_46/while/Identity_3:output:0"=
lstm_46_while_identity_4!lstm_46/while/Identity_4:output:0"=
lstm_46_while_identity_5!lstm_46/while/Identity_5:output:0"P
%lstm_46_while_lstm_46_strided_slice_1'lstm_46_while_lstm_46_strided_slice_1_0"|
;lstm_46_while_lstm_cell_184_biasadd_readvariableop_resource=lstm_46_while_lstm_cell_184_biasadd_readvariableop_resource_0"~
<lstm_46_while_lstm_cell_184_matmul_1_readvariableop_resource>lstm_46_while_lstm_cell_184_matmul_1_readvariableop_resource_0"z
:lstm_46_while_lstm_cell_184_matmul_readvariableop_resource<lstm_46_while_lstm_cell_184_matmul_readvariableop_resource_0"È
alstm_46_while_tensorarrayv2read_tensorlistgetitem_lstm_46_tensorarrayunstack_tensorlistfromtensorclstm_46_while_tensorarrayv2read_tensorlistgetitem_lstm_46_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2h
2lstm_46/while/lstm_cell_184/BiasAdd/ReadVariableOp2lstm_46/while/lstm_cell_184/BiasAdd/ReadVariableOp2f
1lstm_46/while/lstm_cell_184/MatMul/ReadVariableOp1lstm_46/while/lstm_cell_184/MatMul/ReadVariableOp2j
3lstm_46/while/lstm_cell_184/MatMul_1/ReadVariableOp3lstm_46/while/lstm_cell_184/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
º
È
while_cond_3534748
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3534748___redundant_placeholder05
1while_while_cond_3534748___redundant_placeholder15
1while_while_cond_3534748___redundant_placeholder25
1while_while_cond_3534748___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:


è
lstm_46_while_cond_3533681,
(lstm_46_while_lstm_46_while_loop_counter2
.lstm_46_while_lstm_46_while_maximum_iterations
lstm_46_while_placeholder
lstm_46_while_placeholder_1
lstm_46_while_placeholder_2
lstm_46_while_placeholder_3.
*lstm_46_while_less_lstm_46_strided_slice_1E
Alstm_46_while_lstm_46_while_cond_3533681___redundant_placeholder0E
Alstm_46_while_lstm_46_while_cond_3533681___redundant_placeholder1E
Alstm_46_while_lstm_46_while_cond_3533681___redundant_placeholder2E
Alstm_46_while_lstm_46_while_cond_3533681___redundant_placeholder3
lstm_46_while_identity

lstm_46/while/LessLesslstm_46_while_placeholder*lstm_46_while_less_lstm_46_strided_slice_1*
T0*
_output_shapes
: [
lstm_46/while/IdentityIdentitylstm_46/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_46_while_identitylstm_46/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
³Q
û
'rnn_model_21_lstm_46_while_body_3530898F
Brnn_model_21_lstm_46_while_rnn_model_21_lstm_46_while_loop_counterL
Hrnn_model_21_lstm_46_while_rnn_model_21_lstm_46_while_maximum_iterations*
&rnn_model_21_lstm_46_while_placeholder,
(rnn_model_21_lstm_46_while_placeholder_1,
(rnn_model_21_lstm_46_while_placeholder_2,
(rnn_model_21_lstm_46_while_placeholder_3E
Arnn_model_21_lstm_46_while_rnn_model_21_lstm_46_strided_slice_1_0
}rnn_model_21_lstm_46_while_tensorarrayv2read_tensorlistgetitem_rnn_model_21_lstm_46_tensorarrayunstack_tensorlistfromtensor_0\
Irnn_model_21_lstm_46_while_lstm_cell_184_matmul_readvariableop_resource_0:	 ^
Krnn_model_21_lstm_46_while_lstm_cell_184_matmul_1_readvariableop_resource_0:	 Y
Jrnn_model_21_lstm_46_while_lstm_cell_184_biasadd_readvariableop_resource_0:	'
#rnn_model_21_lstm_46_while_identity)
%rnn_model_21_lstm_46_while_identity_1)
%rnn_model_21_lstm_46_while_identity_2)
%rnn_model_21_lstm_46_while_identity_3)
%rnn_model_21_lstm_46_while_identity_4)
%rnn_model_21_lstm_46_while_identity_5C
?rnn_model_21_lstm_46_while_rnn_model_21_lstm_46_strided_slice_1
{rnn_model_21_lstm_46_while_tensorarrayv2read_tensorlistgetitem_rnn_model_21_lstm_46_tensorarrayunstack_tensorlistfromtensorZ
Grnn_model_21_lstm_46_while_lstm_cell_184_matmul_readvariableop_resource:	 \
Irnn_model_21_lstm_46_while_lstm_cell_184_matmul_1_readvariableop_resource:	 W
Hrnn_model_21_lstm_46_while_lstm_cell_184_biasadd_readvariableop_resource:	¢?rnn_model_21/lstm_46/while/lstm_cell_184/BiasAdd/ReadVariableOp¢>rnn_model_21/lstm_46/while/lstm_cell_184/MatMul/ReadVariableOp¢@rnn_model_21/lstm_46/while/lstm_cell_184/MatMul_1/ReadVariableOp
Lrnn_model_21/lstm_46/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    
>rnn_model_21/lstm_46/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}rnn_model_21_lstm_46_while_tensorarrayv2read_tensorlistgetitem_rnn_model_21_lstm_46_tensorarrayunstack_tensorlistfromtensor_0&rnn_model_21_lstm_46_while_placeholderUrnn_model_21/lstm_46/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0É
>rnn_model_21/lstm_46/while/lstm_cell_184/MatMul/ReadVariableOpReadVariableOpIrnn_model_21_lstm_46_while_lstm_cell_184_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype0û
/rnn_model_21/lstm_46/while/lstm_cell_184/MatMulMatMulErnn_model_21/lstm_46/while/TensorArrayV2Read/TensorListGetItem:item:0Frnn_model_21/lstm_46/while/lstm_cell_184/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ
@rnn_model_21/lstm_46/while/lstm_cell_184/MatMul_1/ReadVariableOpReadVariableOpKrnn_model_21_lstm_46_while_lstm_cell_184_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0â
1rnn_model_21/lstm_46/while/lstm_cell_184/MatMul_1MatMul(rnn_model_21_lstm_46_while_placeholder_2Hrnn_model_21/lstm_46/while/lstm_cell_184/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
,rnn_model_21/lstm_46/while/lstm_cell_184/addAddV29rnn_model_21/lstm_46/while/lstm_cell_184/MatMul:product:0;rnn_model_21/lstm_46/while/lstm_cell_184/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ
?rnn_model_21/lstm_46/while/lstm_cell_184/BiasAdd/ReadVariableOpReadVariableOpJrnn_model_21_lstm_46_while_lstm_cell_184_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0é
0rnn_model_21/lstm_46/while/lstm_cell_184/BiasAddBiasAdd0rnn_model_21/lstm_46/while/lstm_cell_184/add:z:0Grnn_model_21/lstm_46/while/lstm_cell_184/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
8rnn_model_21/lstm_46/while/lstm_cell_184/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :±
.rnn_model_21/lstm_46/while/lstm_cell_184/splitSplitArnn_model_21/lstm_46/while/lstm_cell_184/split/split_dim:output:09rnn_model_21/lstm_46/while/lstm_cell_184/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split¦
0rnn_model_21/lstm_46/while/lstm_cell_184/SigmoidSigmoid7rnn_model_21/lstm_46/while/lstm_cell_184/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¨
2rnn_model_21/lstm_46/while/lstm_cell_184/Sigmoid_1Sigmoid7rnn_model_21/lstm_46/while/lstm_cell_184/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ç
,rnn_model_21/lstm_46/while/lstm_cell_184/mulMul6rnn_model_21/lstm_46/while/lstm_cell_184/Sigmoid_1:y:0(rnn_model_21_lstm_46_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
-rnn_model_21/lstm_46/while/lstm_cell_184/ReluRelu7rnn_model_21/lstm_46/while/lstm_cell_184/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ú
.rnn_model_21/lstm_46/while/lstm_cell_184/mul_1Mul4rnn_model_21/lstm_46/while/lstm_cell_184/Sigmoid:y:0;rnn_model_21/lstm_46/while/lstm_cell_184/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ï
.rnn_model_21/lstm_46/while/lstm_cell_184/add_1AddV20rnn_model_21/lstm_46/while/lstm_cell_184/mul:z:02rnn_model_21/lstm_46/while/lstm_cell_184/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¨
2rnn_model_21/lstm_46/while/lstm_cell_184/Sigmoid_2Sigmoid7rnn_model_21/lstm_46/while/lstm_cell_184/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
/rnn_model_21/lstm_46/while/lstm_cell_184/Relu_1Relu2rnn_model_21/lstm_46/while/lstm_cell_184/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Þ
.rnn_model_21/lstm_46/while/lstm_cell_184/mul_2Mul6rnn_model_21/lstm_46/while/lstm_cell_184/Sigmoid_2:y:0=rnn_model_21/lstm_46/while/lstm_cell_184/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
?rnn_model_21/lstm_46/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(rnn_model_21_lstm_46_while_placeholder_1&rnn_model_21_lstm_46_while_placeholder2rnn_model_21/lstm_46/while/lstm_cell_184/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒb
 rnn_model_21/lstm_46/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
rnn_model_21/lstm_46/while/addAddV2&rnn_model_21_lstm_46_while_placeholder)rnn_model_21/lstm_46/while/add/y:output:0*
T0*
_output_shapes
: d
"rnn_model_21/lstm_46/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :»
 rnn_model_21/lstm_46/while/add_1AddV2Brnn_model_21_lstm_46_while_rnn_model_21_lstm_46_while_loop_counter+rnn_model_21/lstm_46/while/add_1/y:output:0*
T0*
_output_shapes
: 
#rnn_model_21/lstm_46/while/IdentityIdentity$rnn_model_21/lstm_46/while/add_1:z:0 ^rnn_model_21/lstm_46/while/NoOp*
T0*
_output_shapes
: ¾
%rnn_model_21/lstm_46/while/Identity_1IdentityHrnn_model_21_lstm_46_while_rnn_model_21_lstm_46_while_maximum_iterations ^rnn_model_21/lstm_46/while/NoOp*
T0*
_output_shapes
: 
%rnn_model_21/lstm_46/while/Identity_2Identity"rnn_model_21/lstm_46/while/add:z:0 ^rnn_model_21/lstm_46/while/NoOp*
T0*
_output_shapes
: Å
%rnn_model_21/lstm_46/while/Identity_3IdentityOrnn_model_21/lstm_46/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^rnn_model_21/lstm_46/while/NoOp*
T0*
_output_shapes
: ¹
%rnn_model_21/lstm_46/while/Identity_4Identity2rnn_model_21/lstm_46/while/lstm_cell_184/mul_2:z:0 ^rnn_model_21/lstm_46/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¹
%rnn_model_21/lstm_46/while/Identity_5Identity2rnn_model_21/lstm_46/while/lstm_cell_184/add_1:z:0 ^rnn_model_21/lstm_46/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ §
rnn_model_21/lstm_46/while/NoOpNoOp@^rnn_model_21/lstm_46/while/lstm_cell_184/BiasAdd/ReadVariableOp?^rnn_model_21/lstm_46/while/lstm_cell_184/MatMul/ReadVariableOpA^rnn_model_21/lstm_46/while/lstm_cell_184/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "S
#rnn_model_21_lstm_46_while_identity,rnn_model_21/lstm_46/while/Identity:output:0"W
%rnn_model_21_lstm_46_while_identity_1.rnn_model_21/lstm_46/while/Identity_1:output:0"W
%rnn_model_21_lstm_46_while_identity_2.rnn_model_21/lstm_46/while/Identity_2:output:0"W
%rnn_model_21_lstm_46_while_identity_3.rnn_model_21/lstm_46/while/Identity_3:output:0"W
%rnn_model_21_lstm_46_while_identity_4.rnn_model_21/lstm_46/while/Identity_4:output:0"W
%rnn_model_21_lstm_46_while_identity_5.rnn_model_21/lstm_46/while/Identity_5:output:0"
Hrnn_model_21_lstm_46_while_lstm_cell_184_biasadd_readvariableop_resourceJrnn_model_21_lstm_46_while_lstm_cell_184_biasadd_readvariableop_resource_0"
Irnn_model_21_lstm_46_while_lstm_cell_184_matmul_1_readvariableop_resourceKrnn_model_21_lstm_46_while_lstm_cell_184_matmul_1_readvariableop_resource_0"
Grnn_model_21_lstm_46_while_lstm_cell_184_matmul_readvariableop_resourceIrnn_model_21_lstm_46_while_lstm_cell_184_matmul_readvariableop_resource_0"
?rnn_model_21_lstm_46_while_rnn_model_21_lstm_46_strided_slice_1Arnn_model_21_lstm_46_while_rnn_model_21_lstm_46_strided_slice_1_0"ü
{rnn_model_21_lstm_46_while_tensorarrayv2read_tensorlistgetitem_rnn_model_21_lstm_46_tensorarrayunstack_tensorlistfromtensor}rnn_model_21_lstm_46_while_tensorarrayv2read_tensorlistgetitem_rnn_model_21_lstm_46_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2
?rnn_model_21/lstm_46/while/lstm_cell_184/BiasAdd/ReadVariableOp?rnn_model_21/lstm_46/while/lstm_cell_184/BiasAdd/ReadVariableOp2
>rnn_model_21/lstm_46/while/lstm_cell_184/MatMul/ReadVariableOp>rnn_model_21/lstm_46/while/lstm_cell_184/MatMul/ReadVariableOp2
@rnn_model_21/lstm_46/while/lstm_cell_184/MatMul_1/ReadVariableOp@rnn_model_21/lstm_46/while/lstm_cell_184/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
¸


%__inference_signature_wrapper_3533430
input_1
unknown:	
	unknown_0:	 
	unknown_1:	
	unknown_2:	 
	unknown_3:	 
	unknown_4:	
	unknown_5:	 
	unknown_6:	 
	unknown_7:	
	unknown_8: 
	unknown_9:
identity¢StatefulPartitionedCall±
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_3531138s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1
ô
è
I__inference_rnn_model_21_layer_call_and_return_conditional_losses_3532683
x"
lstm_45_3532342:	"
lstm_45_3532344:	 
lstm_45_3532346:	"
lstm_46_3532492:	 "
lstm_46_3532494:	 
lstm_46_3532496:	"
lstm_47_3532644:	 "
lstm_47_3532646:	 
lstm_47_3532648:	"
dense_21_3532662: 
dense_21_3532664:
identity¢ dense_21/StatefulPartitionedCall¢lstm_45/StatefulPartitionedCall¢lstm_46/StatefulPartitionedCall¢lstm_47/StatefulPartitionedCall
lstm_45/StatefulPartitionedCallStatefulPartitionedCallxlstm_45_3532342lstm_45_3532344lstm_45_3532346*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_45_layer_call_and_return_conditional_losses_3532341¨
lstm_46/StatefulPartitionedCallStatefulPartitionedCall(lstm_45/StatefulPartitionedCall:output:0lstm_46_3532492lstm_46_3532494lstm_46_3532496*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_46_layer_call_and_return_conditional_losses_3532491¤
lstm_47/StatefulPartitionedCallStatefulPartitionedCall(lstm_46/StatefulPartitionedCall:output:0lstm_47_3532644lstm_47_3532646lstm_47_3532648*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_47_layer_call_and_return_conditional_losses_3532643
 dense_21/StatefulPartitionedCallStatefulPartitionedCall(lstm_47/StatefulPartitionedCall:output:0dense_21_3532662dense_21_3532664*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_3532661â
reshape_8/PartitionedCallPartitionedCall)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reshape_8_layer_call_and_return_conditional_losses_3532680u
IdentityIdentity"reshape_8/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿÏ
NoOpNoOp!^dense_21/StatefulPartitionedCall ^lstm_45/StatefulPartitionedCall ^lstm_46/StatefulPartitionedCall ^lstm_47/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : 2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2B
lstm_45/StatefulPartitionedCalllstm_45/StatefulPartitionedCall2B
lstm_46/StatefulPartitionedCalllstm_46/StatefulPartitionedCall2B
lstm_47/StatefulPartitionedCalllstm_47/StatefulPartitionedCall:N J
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


_user_specified_namex
º
È
while_cond_3532256
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3532256___redundant_placeholder05
1while_while_cond_3532256___redundant_placeholder15
1while_while_cond_3532256___redundant_placeholder25
1while_while_cond_3532256___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
ß

J__inference_lstm_cell_183_layer_call_and_return_conditional_losses_3536319

inputs
states_0
states_11
matmul_readvariableop_resource:	3
 matmul_1_readvariableop_resource:	 .
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ N
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/1
Ñ8
Ú
while_body_3535508
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_184_matmul_readvariableop_resource_0:	 I
6while_lstm_cell_184_matmul_1_readvariableop_resource_0:	 D
5while_lstm_cell_184_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_184_matmul_readvariableop_resource:	 G
4while_lstm_cell_184_matmul_1_readvariableop_resource:	 B
3while_lstm_cell_184_biasadd_readvariableop_resource:	¢*while/lstm_cell_184/BiasAdd/ReadVariableOp¢)while/lstm_cell_184/MatMul/ReadVariableOp¢+while/lstm_cell_184/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0
)while/lstm_cell_184/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_184_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype0¼
while/lstm_cell_184/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_184/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
+while/lstm_cell_184/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_184_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0£
while/lstm_cell_184/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_184/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
while/lstm_cell_184/addAddV2$while/lstm_cell_184/MatMul:product:0&while/lstm_cell_184/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*while/lstm_cell_184/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_184_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0ª
while/lstm_cell_184/BiasAddBiasAddwhile/lstm_cell_184/add:z:02while/lstm_cell_184/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
#while/lstm_cell_184/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ò
while/lstm_cell_184/splitSplit,while/lstm_cell_184/split/split_dim:output:0$while/lstm_cell_184/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split|
while/lstm_cell_184/SigmoidSigmoid"while/lstm_cell_184/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
while/lstm_cell_184/Sigmoid_1Sigmoid"while/lstm_cell_184/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_184/mulMul!while/lstm_cell_184/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ v
while/lstm_cell_184/ReluRelu"while/lstm_cell_184/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_184/mul_1Mulwhile/lstm_cell_184/Sigmoid:y:0&while/lstm_cell_184/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_184/add_1AddV2while/lstm_cell_184/mul:z:0while/lstm_cell_184/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
while/lstm_cell_184/Sigmoid_2Sigmoid"while/lstm_cell_184/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ s
while/lstm_cell_184/Relu_1Reluwhile/lstm_cell_184/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_184/mul_2Mul!while/lstm_cell_184/Sigmoid_2:y:0(while/lstm_cell_184/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Æ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_184/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_184/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
while/Identity_5Identitywhile/lstm_cell_184/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ó

while/NoOpNoOp+^while/lstm_cell_184/BiasAdd/ReadVariableOp*^while/lstm_cell_184/MatMul/ReadVariableOp,^while/lstm_cell_184/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_184_biasadd_readvariableop_resource5while_lstm_cell_184_biasadd_readvariableop_resource_0"n
4while_lstm_cell_184_matmul_1_readvariableop_resource6while_lstm_cell_184_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_184_matmul_readvariableop_resource4while_lstm_cell_184_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2X
*while/lstm_cell_184/BiasAdd/ReadVariableOp*while/lstm_cell_184/BiasAdd/ReadVariableOp2V
)while/lstm_cell_184/MatMul/ReadVariableOp)while/lstm_cell_184/MatMul/ReadVariableOp2Z
+while/lstm_cell_184/MatMul_1/ReadVariableOp+while/lstm_cell_184/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 


è
lstm_45_while_cond_3533980,
(lstm_45_while_lstm_45_while_loop_counter2
.lstm_45_while_lstm_45_while_maximum_iterations
lstm_45_while_placeholder
lstm_45_while_placeholder_1
lstm_45_while_placeholder_2
lstm_45_while_placeholder_3.
*lstm_45_while_less_lstm_45_strided_slice_1E
Alstm_45_while_lstm_45_while_cond_3533980___redundant_placeholder0E
Alstm_45_while_lstm_45_while_cond_3533980___redundant_placeholder1E
Alstm_45_while_lstm_45_while_cond_3533980___redundant_placeholder2E
Alstm_45_while_lstm_45_while_cond_3533980___redundant_placeholder3
lstm_45_while_identity

lstm_45/while/LessLesslstm_45_while_placeholder*lstm_45_while_less_lstm_45_strided_slice_1*
T0*
_output_shapes
: [
lstm_45/while/IdentityIdentitylstm_45/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_45_while_identitylstm_45/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
í9
Ú
while_body_3535696
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_185_matmul_readvariableop_resource_0:	 I
6while_lstm_cell_185_matmul_1_readvariableop_resource_0:	 D
5while_lstm_cell_185_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_185_matmul_readvariableop_resource:	 G
4while_lstm_cell_185_matmul_1_readvariableop_resource:	 B
3while_lstm_cell_185_biasadd_readvariableop_resource:	¢*while/lstm_cell_185/BiasAdd/ReadVariableOp¢)while/lstm_cell_185/MatMul/ReadVariableOp¢+while/lstm_cell_185/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0
)while/lstm_cell_185/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_185_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype0¼
while/lstm_cell_185/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_185/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
+while/lstm_cell_185/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_185_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0£
while/lstm_cell_185/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_185/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
while/lstm_cell_185/addAddV2$while/lstm_cell_185/MatMul:product:0&while/lstm_cell_185/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*while/lstm_cell_185/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_185_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0ª
while/lstm_cell_185/BiasAddBiasAddwhile/lstm_cell_185/add:z:02while/lstm_cell_185/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
#while/lstm_cell_185/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ò
while/lstm_cell_185/splitSplit,while/lstm_cell_185/split/split_dim:output:0$while/lstm_cell_185/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split|
while/lstm_cell_185/SigmoidSigmoid"while/lstm_cell_185/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
while/lstm_cell_185/Sigmoid_1Sigmoid"while/lstm_cell_185/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_185/mulMul!while/lstm_cell_185/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ v
while/lstm_cell_185/ReluRelu"while/lstm_cell_185/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_185/mul_1Mulwhile/lstm_cell_185/Sigmoid:y:0&while/lstm_cell_185/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_185/add_1AddV2while/lstm_cell_185/mul:z:0while/lstm_cell_185/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
while/lstm_cell_185/Sigmoid_2Sigmoid"while/lstm_cell_185/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ s
while/lstm_cell_185/Relu_1Reluwhile/lstm_cell_185/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_185/mul_2Mul!while/lstm_cell_185/Sigmoid_2:y:0(while/lstm_cell_185/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : î
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_185/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_185/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
while/Identity_5Identitywhile/lstm_cell_185/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ó

while/NoOpNoOp+^while/lstm_cell_185/BiasAdd/ReadVariableOp*^while/lstm_cell_185/MatMul/ReadVariableOp,^while/lstm_cell_185/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_185_biasadd_readvariableop_resource5while_lstm_cell_185_biasadd_readvariableop_resource_0"n
4while_lstm_cell_185_matmul_1_readvariableop_resource6while_lstm_cell_185_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_185_matmul_readvariableop_resource4while_lstm_cell_185_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2X
*while/lstm_cell_185/BiasAdd/ReadVariableOp*while/lstm_cell_185/BiasAdd/ReadVariableOp2V
)while/lstm_cell_185/MatMul/ReadVariableOp)while/lstm_cell_185/MatMul/ReadVariableOp2Z
+while/lstm_cell_185/MatMul_1/ReadVariableOp+while/lstm_cell_185/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
`

 __inference__traced_save_3536696
file_prefixH
Dsavev2_rnn_model_21_lstm_45_lstm_cell_183_kernel_read_readvariableopR
Nsavev2_rnn_model_21_lstm_45_lstm_cell_183_recurrent_kernel_read_readvariableopF
Bsavev2_rnn_model_21_lstm_45_lstm_cell_183_bias_read_readvariableopH
Dsavev2_rnn_model_21_lstm_46_lstm_cell_184_kernel_read_readvariableopR
Nsavev2_rnn_model_21_lstm_46_lstm_cell_184_recurrent_kernel_read_readvariableopF
Bsavev2_rnn_model_21_lstm_46_lstm_cell_184_bias_read_readvariableopH
Dsavev2_rnn_model_21_lstm_47_lstm_cell_185_kernel_read_readvariableopR
Nsavev2_rnn_model_21_lstm_47_lstm_cell_185_recurrent_kernel_read_readvariableopF
Bsavev2_rnn_model_21_lstm_47_lstm_cell_185_bias_read_readvariableop;
7savev2_rnn_model_21_dense_21_kernel_read_readvariableop9
5savev2_rnn_model_21_dense_21_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopO
Ksavev2_adam_rnn_model_21_lstm_45_lstm_cell_183_kernel_m_read_readvariableopY
Usavev2_adam_rnn_model_21_lstm_45_lstm_cell_183_recurrent_kernel_m_read_readvariableopM
Isavev2_adam_rnn_model_21_lstm_45_lstm_cell_183_bias_m_read_readvariableopO
Ksavev2_adam_rnn_model_21_lstm_46_lstm_cell_184_kernel_m_read_readvariableopY
Usavev2_adam_rnn_model_21_lstm_46_lstm_cell_184_recurrent_kernel_m_read_readvariableopM
Isavev2_adam_rnn_model_21_lstm_46_lstm_cell_184_bias_m_read_readvariableopO
Ksavev2_adam_rnn_model_21_lstm_47_lstm_cell_185_kernel_m_read_readvariableopY
Usavev2_adam_rnn_model_21_lstm_47_lstm_cell_185_recurrent_kernel_m_read_readvariableopM
Isavev2_adam_rnn_model_21_lstm_47_lstm_cell_185_bias_m_read_readvariableopB
>savev2_adam_rnn_model_21_dense_21_kernel_m_read_readvariableop@
<savev2_adam_rnn_model_21_dense_21_bias_m_read_readvariableopO
Ksavev2_adam_rnn_model_21_lstm_45_lstm_cell_183_kernel_v_read_readvariableopY
Usavev2_adam_rnn_model_21_lstm_45_lstm_cell_183_recurrent_kernel_v_read_readvariableopM
Isavev2_adam_rnn_model_21_lstm_45_lstm_cell_183_bias_v_read_readvariableopO
Ksavev2_adam_rnn_model_21_lstm_46_lstm_cell_184_kernel_v_read_readvariableopY
Usavev2_adam_rnn_model_21_lstm_46_lstm_cell_184_recurrent_kernel_v_read_readvariableopM
Isavev2_adam_rnn_model_21_lstm_46_lstm_cell_184_bias_v_read_readvariableopO
Ksavev2_adam_rnn_model_21_lstm_47_lstm_cell_185_kernel_v_read_readvariableopY
Usavev2_adam_rnn_model_21_lstm_47_lstm_cell_185_recurrent_kernel_v_read_readvariableopM
Isavev2_adam_rnn_model_21_lstm_47_lstm_cell_185_bias_v_read_readvariableopB
>savev2_adam_rnn_model_21_dense_21_kernel_v_read_readvariableop@
<savev2_adam_rnn_model_21_dense_21_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ú
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*
valueùBö+B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÃ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Æ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Dsavev2_rnn_model_21_lstm_45_lstm_cell_183_kernel_read_readvariableopNsavev2_rnn_model_21_lstm_45_lstm_cell_183_recurrent_kernel_read_readvariableopBsavev2_rnn_model_21_lstm_45_lstm_cell_183_bias_read_readvariableopDsavev2_rnn_model_21_lstm_46_lstm_cell_184_kernel_read_readvariableopNsavev2_rnn_model_21_lstm_46_lstm_cell_184_recurrent_kernel_read_readvariableopBsavev2_rnn_model_21_lstm_46_lstm_cell_184_bias_read_readvariableopDsavev2_rnn_model_21_lstm_47_lstm_cell_185_kernel_read_readvariableopNsavev2_rnn_model_21_lstm_47_lstm_cell_185_recurrent_kernel_read_readvariableopBsavev2_rnn_model_21_lstm_47_lstm_cell_185_bias_read_readvariableop7savev2_rnn_model_21_dense_21_kernel_read_readvariableop5savev2_rnn_model_21_dense_21_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopKsavev2_adam_rnn_model_21_lstm_45_lstm_cell_183_kernel_m_read_readvariableopUsavev2_adam_rnn_model_21_lstm_45_lstm_cell_183_recurrent_kernel_m_read_readvariableopIsavev2_adam_rnn_model_21_lstm_45_lstm_cell_183_bias_m_read_readvariableopKsavev2_adam_rnn_model_21_lstm_46_lstm_cell_184_kernel_m_read_readvariableopUsavev2_adam_rnn_model_21_lstm_46_lstm_cell_184_recurrent_kernel_m_read_readvariableopIsavev2_adam_rnn_model_21_lstm_46_lstm_cell_184_bias_m_read_readvariableopKsavev2_adam_rnn_model_21_lstm_47_lstm_cell_185_kernel_m_read_readvariableopUsavev2_adam_rnn_model_21_lstm_47_lstm_cell_185_recurrent_kernel_m_read_readvariableopIsavev2_adam_rnn_model_21_lstm_47_lstm_cell_185_bias_m_read_readvariableop>savev2_adam_rnn_model_21_dense_21_kernel_m_read_readvariableop<savev2_adam_rnn_model_21_dense_21_bias_m_read_readvariableopKsavev2_adam_rnn_model_21_lstm_45_lstm_cell_183_kernel_v_read_readvariableopUsavev2_adam_rnn_model_21_lstm_45_lstm_cell_183_recurrent_kernel_v_read_readvariableopIsavev2_adam_rnn_model_21_lstm_45_lstm_cell_183_bias_v_read_readvariableopKsavev2_adam_rnn_model_21_lstm_46_lstm_cell_184_kernel_v_read_readvariableopUsavev2_adam_rnn_model_21_lstm_46_lstm_cell_184_recurrent_kernel_v_read_readvariableopIsavev2_adam_rnn_model_21_lstm_46_lstm_cell_184_bias_v_read_readvariableopKsavev2_adam_rnn_model_21_lstm_47_lstm_cell_185_kernel_v_read_readvariableopUsavev2_adam_rnn_model_21_lstm_47_lstm_cell_185_recurrent_kernel_v_read_readvariableopIsavev2_adam_rnn_model_21_lstm_47_lstm_cell_185_bias_v_read_readvariableop>savev2_adam_rnn_model_21_dense_21_kernel_v_read_readvariableop<savev2_adam_rnn_model_21_dense_21_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *9
dtypes/
-2+	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*à
_input_shapesÎ
Ë: :	:	 ::	 :	 ::	 :	 :: :: : : : : : : : : :	:	 ::	 :	 ::	 :	 :: ::	:	 ::	 :	 ::	 :	 :: :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	:%!

_output_shapes
:	 :!

_output_shapes	
::%!

_output_shapes
:	 :%!

_output_shapes
:	 :!

_output_shapes	
::%!

_output_shapes
:	 :%!

_output_shapes
:	 :!	

_output_shapes	
::$
 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:%!

_output_shapes
:	 :!

_output_shapes	
::%!

_output_shapes
:	 :%!

_output_shapes
:	 :!

_output_shapes	
::%!

_output_shapes
:	 :%!

_output_shapes
:	 :!

_output_shapes	
::$ 

_output_shapes

: : 

_output_shapes
::% !

_output_shapes
:	:%!!

_output_shapes
:	 :!"

_output_shapes	
::%#!

_output_shapes
:	 :%$!

_output_shapes
:	 :!%

_output_shapes	
::%&!

_output_shapes
:	 :%'!

_output_shapes
:	 :!(

_output_shapes	
::$) 

_output_shapes

: : *

_output_shapes
::+

_output_shapes
: 
í9
Ú
while_body_3535841
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_185_matmul_readvariableop_resource_0:	 I
6while_lstm_cell_185_matmul_1_readvariableop_resource_0:	 D
5while_lstm_cell_185_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_185_matmul_readvariableop_resource:	 G
4while_lstm_cell_185_matmul_1_readvariableop_resource:	 B
3while_lstm_cell_185_biasadd_readvariableop_resource:	¢*while/lstm_cell_185/BiasAdd/ReadVariableOp¢)while/lstm_cell_185/MatMul/ReadVariableOp¢+while/lstm_cell_185/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0
)while/lstm_cell_185/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_185_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype0¼
while/lstm_cell_185/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_185/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
+while/lstm_cell_185/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_185_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0£
while/lstm_cell_185/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_185/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
while/lstm_cell_185/addAddV2$while/lstm_cell_185/MatMul:product:0&while/lstm_cell_185/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*while/lstm_cell_185/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_185_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0ª
while/lstm_cell_185/BiasAddBiasAddwhile/lstm_cell_185/add:z:02while/lstm_cell_185/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
#while/lstm_cell_185/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ò
while/lstm_cell_185/splitSplit,while/lstm_cell_185/split/split_dim:output:0$while/lstm_cell_185/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split|
while/lstm_cell_185/SigmoidSigmoid"while/lstm_cell_185/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
while/lstm_cell_185/Sigmoid_1Sigmoid"while/lstm_cell_185/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_185/mulMul!while/lstm_cell_185/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ v
while/lstm_cell_185/ReluRelu"while/lstm_cell_185/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_185/mul_1Mulwhile/lstm_cell_185/Sigmoid:y:0&while/lstm_cell_185/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_185/add_1AddV2while/lstm_cell_185/mul:z:0while/lstm_cell_185/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
while/lstm_cell_185/Sigmoid_2Sigmoid"while/lstm_cell_185/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ s
while/lstm_cell_185/Relu_1Reluwhile/lstm_cell_185/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_185/mul_2Mul!while/lstm_cell_185/Sigmoid_2:y:0(while/lstm_cell_185/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : î
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_185/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_185/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
while/Identity_5Identitywhile/lstm_cell_185/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ó

while/NoOpNoOp+^while/lstm_cell_185/BiasAdd/ReadVariableOp*^while/lstm_cell_185/MatMul/ReadVariableOp,^while/lstm_cell_185/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_185_biasadd_readvariableop_resource5while_lstm_cell_185_biasadd_readvariableop_resource_0"n
4while_lstm_cell_185_matmul_1_readvariableop_resource6while_lstm_cell_185_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_185_matmul_readvariableop_resource4while_lstm_cell_185_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2X
*while/lstm_cell_185/BiasAdd/ReadVariableOp*while/lstm_cell_185/BiasAdd/ReadVariableOp2V
)while/lstm_cell_185/MatMul/ReadVariableOp)while/lstm_cell_185/MatMul/ReadVariableOp2Z
+while/lstm_cell_185/MatMul_1/ReadVariableOp+while/lstm_cell_185/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
çK
¢
D__inference_lstm_47_layer_call_and_return_conditional_losses_3536071

inputs?
,lstm_cell_185_matmul_readvariableop_resource:	 A
.lstm_cell_185_matmul_1_readvariableop_resource:	 <
-lstm_cell_185_biasadd_readvariableop_resource:	
identity¢$lstm_cell_185/BiasAdd/ReadVariableOp¢#lstm_cell_185/MatMul/ReadVariableOp¢%lstm_cell_185/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿ D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask
#lstm_cell_185/MatMul/ReadVariableOpReadVariableOp,lstm_cell_185_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_185/MatMulMatMulstrided_slice_2:output:0+lstm_cell_185/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%lstm_cell_185/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_185_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_185/MatMul_1MatMulzeros:output:0-lstm_cell_185/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_185/addAddV2lstm_cell_185/MatMul:product:0 lstm_cell_185/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_185/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_185_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_185/BiasAddBiasAddlstm_cell_185/add:z:0,lstm_cell_185/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
lstm_cell_185/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :à
lstm_cell_185/splitSplit&lstm_cell_185/split/split_dim:output:0lstm_cell_185/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitp
lstm_cell_185/SigmoidSigmoidlstm_cell_185/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
lstm_cell_185/Sigmoid_1Sigmoidlstm_cell_185/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y
lstm_cell_185/mulMullstm_cell_185/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
lstm_cell_185/ReluRelulstm_cell_185/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_185/mul_1Mullstm_cell_185/Sigmoid:y:0 lstm_cell_185/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
lstm_cell_185/add_1AddV2lstm_cell_185/mul:z:0lstm_cell_185/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
lstm_cell_185/Sigmoid_2Sigmoidlstm_cell_185/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ g
lstm_cell_185/Relu_1Relulstm_cell_185/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_185/mul_2Mullstm_cell_185/Sigmoid_2:y:0"lstm_cell_185/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_185_matmul_readvariableop_resource.lstm_cell_185_matmul_1_readvariableop_resource-lstm_cell_185_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3535986*
condR
while_cond_3535985*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ã
NoOpNoOp%^lstm_cell_185/BiasAdd/ReadVariableOp$^lstm_cell_185/MatMul/ReadVariableOp&^lstm_cell_185/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
 : : : 2L
$lstm_cell_185/BiasAdd/ReadVariableOp$lstm_cell_185/BiasAdd/ReadVariableOp2J
#lstm_cell_185/MatMul/ReadVariableOp#lstm_cell_185/MatMul/ReadVariableOp2N
%lstm_cell_185/MatMul_1/ReadVariableOp%lstm_cell_185/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
 
_user_specified_nameinputs


è
lstm_47_while_cond_3534259,
(lstm_47_while_lstm_47_while_loop_counter2
.lstm_47_while_lstm_47_while_maximum_iterations
lstm_47_while_placeholder
lstm_47_while_placeholder_1
lstm_47_while_placeholder_2
lstm_47_while_placeholder_3.
*lstm_47_while_less_lstm_47_strided_slice_1E
Alstm_47_while_lstm_47_while_cond_3534259___redundant_placeholder0E
Alstm_47_while_lstm_47_while_cond_3534259___redundant_placeholder1E
Alstm_47_while_lstm_47_while_cond_3534259___redundant_placeholder2E
Alstm_47_while_lstm_47_while_cond_3534259___redundant_placeholder3
lstm_47_while_identity

lstm_47/while/LessLesslstm_47_while_placeholder*lstm_47_while_less_lstm_47_strided_slice_1*
T0*
_output_shapes
: [
lstm_47/while/IdentityIdentitylstm_47/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_47_while_identitylstm_47/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
×

J__inference_lstm_cell_183_layer_call_and_return_conditional_losses_3531351

inputs

states
states_11
matmul_readvariableop_resource:	3
 matmul_1_readvariableop_resource:	 .
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ N
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates
ù
¶
)__inference_lstm_47_layer_call_fn_3535625

inputs
unknown:	 
	unknown_0:	 
	unknown_1:	
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_47_layer_call_and_return_conditional_losses_3532643o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
 : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
 
_user_specified_nameinputs
«
¸
)__inference_lstm_46_layer_call_fn_3534987
inputs_0
unknown:	 
	unknown_0:	 
	unknown_1:	
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_46_layer_call_and_return_conditional_losses_3531638|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0
Ö


.__inference_rnn_model_21_layer_call_fn_3533484
x
unknown:	
	unknown_0:	 
	unknown_1:	
	unknown_2:	 
	unknown_3:	 
	unknown_4:	
	unknown_5:	 
	unknown_6:	 
	unknown_7:	
	unknown_8: 
	unknown_9:
identity¢StatefulPartitionedCallÒ
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_rnn_model_21_layer_call_and_return_conditional_losses_3533281s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


_user_specified_namex
#
ñ
while_body_3531760
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_184_3531784_0:	 0
while_lstm_cell_184_3531786_0:	 ,
while_lstm_cell_184_3531788_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_184_3531784:	 .
while_lstm_cell_184_3531786:	 *
while_lstm_cell_184_3531788:	¢+while/lstm_cell_184/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0¼
+while/lstm_cell_184/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_184_3531784_0while_lstm_cell_184_3531786_0while_lstm_cell_184_3531788_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_184_layer_call_and_return_conditional_losses_3531701Ý
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder4while/lstm_cell_184/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity4while/lstm_cell_184/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/Identity_5Identity4while/lstm_cell_184/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z

while/NoOpNoOp,^while/lstm_cell_184/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"<
while_lstm_cell_184_3531784while_lstm_cell_184_3531784_0"<
while_lstm_cell_184_3531786while_lstm_cell_184_3531786_0"<
while_lstm_cell_184_3531788while_lstm_cell_184_3531788_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2Z
+while/lstm_cell_184/StatefulPartitionedCall+while/lstm_cell_184/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 

ì
'rnn_model_21_lstm_47_while_cond_3531037F
Brnn_model_21_lstm_47_while_rnn_model_21_lstm_47_while_loop_counterL
Hrnn_model_21_lstm_47_while_rnn_model_21_lstm_47_while_maximum_iterations*
&rnn_model_21_lstm_47_while_placeholder,
(rnn_model_21_lstm_47_while_placeholder_1,
(rnn_model_21_lstm_47_while_placeholder_2,
(rnn_model_21_lstm_47_while_placeholder_3H
Drnn_model_21_lstm_47_while_less_rnn_model_21_lstm_47_strided_slice_1_
[rnn_model_21_lstm_47_while_rnn_model_21_lstm_47_while_cond_3531037___redundant_placeholder0_
[rnn_model_21_lstm_47_while_rnn_model_21_lstm_47_while_cond_3531037___redundant_placeholder1_
[rnn_model_21_lstm_47_while_rnn_model_21_lstm_47_while_cond_3531037___redundant_placeholder2_
[rnn_model_21_lstm_47_while_rnn_model_21_lstm_47_while_cond_3531037___redundant_placeholder3'
#rnn_model_21_lstm_47_while_identity
¶
rnn_model_21/lstm_47/while/LessLess&rnn_model_21_lstm_47_while_placeholderDrnn_model_21_lstm_47_while_less_rnn_model_21_lstm_47_strided_slice_1*
T0*
_output_shapes
: u
#rnn_model_21/lstm_47/while/IdentityIdentity#rnn_model_21/lstm_47/while/Less:z:0*
T0
*
_output_shapes
: "S
#rnn_model_21_lstm_47_while_identity,rnn_model_21/lstm_47/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
×

J__inference_lstm_cell_183_layer_call_and_return_conditional_losses_3531205

inputs

states
states_11
matmul_readvariableop_resource:	3
 matmul_1_readvariableop_resource:	 .
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ N
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates
º
È
while_cond_3532962
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3532962___redundant_placeholder05
1while_while_cond_3532962___redundant_placeholder15
1while_while_cond_3532962___redundant_placeholder25
1while_while_cond_3532962___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
º
È
while_cond_3531568
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3531568___redundant_placeholder05
1while_while_cond_3531568___redundant_placeholder15
1while_while_cond_3531568___redundant_placeholder25
1while_while_cond_3531568___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
º
È
while_cond_3532557
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3532557___redundant_placeholder05
1while_while_cond_3532557___redundant_placeholder15
1while_while_cond_3532557___redundant_placeholder25
1while_while_cond_3532557___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:


è
lstm_46_while_cond_3534119,
(lstm_46_while_lstm_46_while_loop_counter2
.lstm_46_while_lstm_46_while_maximum_iterations
lstm_46_while_placeholder
lstm_46_while_placeholder_1
lstm_46_while_placeholder_2
lstm_46_while_placeholder_3.
*lstm_46_while_less_lstm_46_strided_slice_1E
Alstm_46_while_lstm_46_while_cond_3534119___redundant_placeholder0E
Alstm_46_while_lstm_46_while_cond_3534119___redundant_placeholder1E
Alstm_46_while_lstm_46_while_cond_3534119___redundant_placeholder2E
Alstm_46_while_lstm_46_while_cond_3534119___redundant_placeholder3
lstm_46_while_identity

lstm_46/while/LessLesslstm_46_while_placeholder*lstm_46_while_less_lstm_46_strided_slice_1*
T0*
_output_shapes
: [
lstm_46/while/IdentityIdentitylstm_46/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_46_while_identitylstm_46/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
Ö


.__inference_rnn_model_21_layer_call_fn_3533457
x
unknown:	
	unknown_0:	 
	unknown_1:	
	unknown_2:	 
	unknown_3:	 
	unknown_4:	
	unknown_5:	 
	unknown_6:	 
	unknown_7:	
	unknown_8: 
	unknown_9:
identity¢StatefulPartitionedCallÒ
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_rnn_model_21_layer_call_and_return_conditional_losses_3532683s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


_user_specified_namex
È	
ö
E__inference_dense_21_layer_call_and_return_conditional_losses_3536235

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
×

J__inference_lstm_cell_185_layer_call_and_return_conditional_losses_3532053

inputs

states
states_11
matmul_readvariableop_resource:	 3
 matmul_1_readvariableop_resource:	 .
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 *
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ N
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates
áJ
¢
D__inference_lstm_46_layer_call_and_return_conditional_losses_3532491

inputs?
,lstm_cell_184_matmul_readvariableop_resource:	 A
.lstm_cell_184_matmul_1_readvariableop_resource:	 <
-lstm_cell_184_biasadd_readvariableop_resource:	
identity¢$lstm_cell_184/BiasAdd/ReadVariableOp¢#lstm_cell_184/MatMul/ReadVariableOp¢%lstm_cell_184/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿ D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask
#lstm_cell_184/MatMul/ReadVariableOpReadVariableOp,lstm_cell_184_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_184/MatMulMatMulstrided_slice_2:output:0+lstm_cell_184/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%lstm_cell_184/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_184_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_184/MatMul_1MatMulzeros:output:0-lstm_cell_184/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_184/addAddV2lstm_cell_184/MatMul:product:0 lstm_cell_184/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_184/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_184_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_184/BiasAddBiasAddlstm_cell_184/add:z:0,lstm_cell_184/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
lstm_cell_184/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :à
lstm_cell_184/splitSplit&lstm_cell_184/split/split_dim:output:0lstm_cell_184/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitp
lstm_cell_184/SigmoidSigmoidlstm_cell_184/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
lstm_cell_184/Sigmoid_1Sigmoidlstm_cell_184/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y
lstm_cell_184/mulMullstm_cell_184/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
lstm_cell_184/ReluRelulstm_cell_184/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_184/mul_1Mullstm_cell_184/Sigmoid:y:0 lstm_cell_184/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
lstm_cell_184/add_1AddV2lstm_cell_184/mul:z:0lstm_cell_184/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
lstm_cell_184/Sigmoid_2Sigmoidlstm_cell_184/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ g
lstm_cell_184/Relu_1Relulstm_cell_184/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_184/mul_2Mullstm_cell_184/Sigmoid_2:y:0"lstm_cell_184/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_184_matmul_readvariableop_resource.lstm_cell_184_matmul_1_readvariableop_resource-lstm_cell_184_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3532407*
condR
while_cond_3532406*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 Ã
NoOpNoOp%^lstm_cell_184/BiasAdd/ReadVariableOp$^lstm_cell_184/MatMul/ReadVariableOp&^lstm_cell_184/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
 : : : 2L
$lstm_cell_184/BiasAdd/ReadVariableOp$lstm_cell_184/BiasAdd/ReadVariableOp2J
#lstm_cell_184/MatMul/ReadVariableOp#lstm_cell_184/MatMul/ReadVariableOp2N
%lstm_cell_184/MatMul_1/ReadVariableOp%lstm_cell_184/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
 
_user_specified_nameinputs
Ñ8
Ú
while_body_3532407
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_184_matmul_readvariableop_resource_0:	 I
6while_lstm_cell_184_matmul_1_readvariableop_resource_0:	 D
5while_lstm_cell_184_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_184_matmul_readvariableop_resource:	 G
4while_lstm_cell_184_matmul_1_readvariableop_resource:	 B
3while_lstm_cell_184_biasadd_readvariableop_resource:	¢*while/lstm_cell_184/BiasAdd/ReadVariableOp¢)while/lstm_cell_184/MatMul/ReadVariableOp¢+while/lstm_cell_184/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0
)while/lstm_cell_184/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_184_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype0¼
while/lstm_cell_184/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_184/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
+while/lstm_cell_184/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_184_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0£
while/lstm_cell_184/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_184/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
while/lstm_cell_184/addAddV2$while/lstm_cell_184/MatMul:product:0&while/lstm_cell_184/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*while/lstm_cell_184/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_184_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0ª
while/lstm_cell_184/BiasAddBiasAddwhile/lstm_cell_184/add:z:02while/lstm_cell_184/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
#while/lstm_cell_184/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ò
while/lstm_cell_184/splitSplit,while/lstm_cell_184/split/split_dim:output:0$while/lstm_cell_184/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split|
while/lstm_cell_184/SigmoidSigmoid"while/lstm_cell_184/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
while/lstm_cell_184/Sigmoid_1Sigmoid"while/lstm_cell_184/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_184/mulMul!while/lstm_cell_184/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ v
while/lstm_cell_184/ReluRelu"while/lstm_cell_184/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_184/mul_1Mulwhile/lstm_cell_184/Sigmoid:y:0&while/lstm_cell_184/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_184/add_1AddV2while/lstm_cell_184/mul:z:0while/lstm_cell_184/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
while/lstm_cell_184/Sigmoid_2Sigmoid"while/lstm_cell_184/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ s
while/lstm_cell_184/Relu_1Reluwhile/lstm_cell_184/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_184/mul_2Mul!while/lstm_cell_184/Sigmoid_2:y:0(while/lstm_cell_184/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Æ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_184/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_184/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
while/Identity_5Identitywhile/lstm_cell_184/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ó

while/NoOpNoOp+^while/lstm_cell_184/BiasAdd/ReadVariableOp*^while/lstm_cell_184/MatMul/ReadVariableOp,^while/lstm_cell_184/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_184_biasadd_readvariableop_resource5while_lstm_cell_184_biasadd_readvariableop_resource_0"n
4while_lstm_cell_184_matmul_1_readvariableop_resource6while_lstm_cell_184_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_184_matmul_readvariableop_resource4while_lstm_cell_184_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2X
*while/lstm_cell_184/BiasAdd/ReadVariableOp*while/lstm_cell_184/BiasAdd/ReadVariableOp2V
)while/lstm_cell_184/MatMul/ReadVariableOp)while/lstm_cell_184/MatMul/ReadVariableOp2Z
+while/lstm_cell_184/MatMul_1/ReadVariableOp+while/lstm_cell_184/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
Ñ8
Ú
while_body_3534749
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_183_matmul_readvariableop_resource_0:	I
6while_lstm_cell_183_matmul_1_readvariableop_resource_0:	 D
5while_lstm_cell_183_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_183_matmul_readvariableop_resource:	G
4while_lstm_cell_183_matmul_1_readvariableop_resource:	 B
3while_lstm_cell_183_biasadd_readvariableop_resource:	¢*while/lstm_cell_183/BiasAdd/ReadVariableOp¢)while/lstm_cell_183/MatMul/ReadVariableOp¢+while/lstm_cell_183/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
)while/lstm_cell_183/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_183_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0¼
while/lstm_cell_183/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_183/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
+while/lstm_cell_183/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_183_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0£
while/lstm_cell_183/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_183/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
while/lstm_cell_183/addAddV2$while/lstm_cell_183/MatMul:product:0&while/lstm_cell_183/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*while/lstm_cell_183/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_183_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0ª
while/lstm_cell_183/BiasAddBiasAddwhile/lstm_cell_183/add:z:02while/lstm_cell_183/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
#while/lstm_cell_183/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ò
while/lstm_cell_183/splitSplit,while/lstm_cell_183/split/split_dim:output:0$while/lstm_cell_183/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split|
while/lstm_cell_183/SigmoidSigmoid"while/lstm_cell_183/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
while/lstm_cell_183/Sigmoid_1Sigmoid"while/lstm_cell_183/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_183/mulMul!while/lstm_cell_183/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ v
while/lstm_cell_183/ReluRelu"while/lstm_cell_183/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_183/mul_1Mulwhile/lstm_cell_183/Sigmoid:y:0&while/lstm_cell_183/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_183/add_1AddV2while/lstm_cell_183/mul:z:0while/lstm_cell_183/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
while/lstm_cell_183/Sigmoid_2Sigmoid"while/lstm_cell_183/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ s
while/lstm_cell_183/Relu_1Reluwhile/lstm_cell_183/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_183/mul_2Mul!while/lstm_cell_183/Sigmoid_2:y:0(while/lstm_cell_183/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Æ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_183/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_183/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
while/Identity_5Identitywhile/lstm_cell_183/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ó

while/NoOpNoOp+^while/lstm_cell_183/BiasAdd/ReadVariableOp*^while/lstm_cell_183/MatMul/ReadVariableOp,^while/lstm_cell_183/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_183_biasadd_readvariableop_resource5while_lstm_cell_183_biasadd_readvariableop_resource_0"n
4while_lstm_cell_183_matmul_1_readvariableop_resource6while_lstm_cell_183_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_183_matmul_readvariableop_resource4while_lstm_cell_183_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2X
*while/lstm_cell_183/BiasAdd/ReadVariableOp*while/lstm_cell_183/BiasAdd/ReadVariableOp2V
)while/lstm_cell_183/MatMul/ReadVariableOp)while/lstm_cell_183/MatMul/ReadVariableOp2Z
+while/lstm_cell_183/MatMul_1/ReadVariableOp+while/lstm_cell_183/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
è


.__inference_rnn_model_21_layer_call_fn_3532708
input_1
unknown:	
	unknown_0:	 
	unknown_1:	
	unknown_2:	 
	unknown_3:	 
	unknown_4:	
	unknown_5:	 
	unknown_6:	 
	unknown_7:	
	unknown_8: 
	unknown_9:
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_rnn_model_21_layer_call_and_return_conditional_losses_3532683s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1
ï
ø
/__inference_lstm_cell_185_layer_call_fn_3536483

inputs
states_0
states_1
unknown:	 
	unknown_0:	 
	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_185_layer_call_and_return_conditional_losses_3532053o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/1

ì
'rnn_model_21_lstm_46_while_cond_3530897F
Brnn_model_21_lstm_46_while_rnn_model_21_lstm_46_while_loop_counterL
Hrnn_model_21_lstm_46_while_rnn_model_21_lstm_46_while_maximum_iterations*
&rnn_model_21_lstm_46_while_placeholder,
(rnn_model_21_lstm_46_while_placeholder_1,
(rnn_model_21_lstm_46_while_placeholder_2,
(rnn_model_21_lstm_46_while_placeholder_3H
Drnn_model_21_lstm_46_while_less_rnn_model_21_lstm_46_strided_slice_1_
[rnn_model_21_lstm_46_while_rnn_model_21_lstm_46_while_cond_3530897___redundant_placeholder0_
[rnn_model_21_lstm_46_while_rnn_model_21_lstm_46_while_cond_3530897___redundant_placeholder1_
[rnn_model_21_lstm_46_while_rnn_model_21_lstm_46_while_cond_3530897___redundant_placeholder2_
[rnn_model_21_lstm_46_while_rnn_model_21_lstm_46_while_cond_3530897___redundant_placeholder3'
#rnn_model_21_lstm_46_while_identity
¶
rnn_model_21/lstm_46/while/LessLess&rnn_model_21_lstm_46_while_placeholderDrnn_model_21_lstm_46_while_less_rnn_model_21_lstm_46_strided_slice_1*
T0*
_output_shapes
: u
#rnn_model_21/lstm_46/while/IdentityIdentity#rnn_model_21/lstm_46/while/Less:z:0*
T0
*
_output_shapes
: "S
#rnn_model_21_lstm_46_while_identity,rnn_model_21/lstm_46/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
º
È
while_cond_3531759
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3531759___redundant_placeholder05
1while_while_cond_3531759___redundant_placeholder15
1while_while_cond_3531759___redundant_placeholder25
1while_while_cond_3531759___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
º
È
while_cond_3536130
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3536130___redundant_placeholder05
1while_while_cond_3536130___redundant_placeholder15
1while_while_cond_3536130___redundant_placeholder25
1while_while_cond_3536130___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
ß

J__inference_lstm_cell_185_layer_call_and_return_conditional_losses_3536547

inputs
states_0
states_11
matmul_readvariableop_resource:	 3
 matmul_1_readvariableop_resource:	 .
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 *
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ N
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/1

¶
)__inference_lstm_45_layer_call_fn_3534393

inputs
unknown:	
	unknown_0:	 
	unknown_1:	
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_45_layer_call_and_return_conditional_losses_3532341s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
#
ñ
while_body_3531569
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_184_3531593_0:	 0
while_lstm_cell_184_3531595_0:	 ,
while_lstm_cell_184_3531597_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_184_3531593:	 .
while_lstm_cell_184_3531595:	 *
while_lstm_cell_184_3531597:	¢+while/lstm_cell_184/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0¼
+while/lstm_cell_184/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_184_3531593_0while_lstm_cell_184_3531595_0while_lstm_cell_184_3531597_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_184_layer_call_and_return_conditional_losses_3531555Ý
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder4while/lstm_cell_184/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity4while/lstm_cell_184/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/Identity_5Identity4while/lstm_cell_184/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z

while/NoOpNoOp,^while/lstm_cell_184/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"<
while_lstm_cell_184_3531593while_lstm_cell_184_3531593_0"<
while_lstm_cell_184_3531595while_lstm_cell_184_3531595_0"<
while_lstm_cell_184_3531597while_lstm_cell_184_3531597_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2Z
+while/lstm_cell_184/StatefulPartitionedCall+while/lstm_cell_184/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
º
È
while_cond_3534462
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3534462___redundant_placeholder05
1while_while_cond_3534462___redundant_placeholder15
1while_while_cond_3534462___redundant_placeholder25
1while_while_cond_3534462___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
º
È
while_cond_3534605
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3534605___redundant_placeholder05
1while_while_cond_3534605___redundant_placeholder15
1while_while_cond_3534605___redundant_placeholder25
1while_while_cond_3534605___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
®8

D__inference_lstm_45_layer_call_and_return_conditional_losses_3531479

inputs(
lstm_cell_183_3531397:	(
lstm_cell_183_3531399:	 $
lstm_cell_183_3531401:	
identity¢%lstm_cell_183/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskþ
%lstm_cell_183/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_183_3531397lstm_cell_183_3531399lstm_cell_183_3531401*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_183_layer_call_and_return_conditional_losses_3531351n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¿
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_183_3531397lstm_cell_183_3531399lstm_cell_183_3531401*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3531410*
condR
while_cond_3531409*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ v
NoOpNoOp&^lstm_cell_183/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2N
%lstm_cell_183/StatefulPartitionedCall%lstm_cell_183/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
©
G
+__inference_reshape_8_layer_call_fn_3536240

inputs
identityµ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reshape_8_layer_call_and_return_conditional_losses_3532680d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ8
Ú
while_body_3532257
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_183_matmul_readvariableop_resource_0:	I
6while_lstm_cell_183_matmul_1_readvariableop_resource_0:	 D
5while_lstm_cell_183_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_183_matmul_readvariableop_resource:	G
4while_lstm_cell_183_matmul_1_readvariableop_resource:	 B
3while_lstm_cell_183_biasadd_readvariableop_resource:	¢*while/lstm_cell_183/BiasAdd/ReadVariableOp¢)while/lstm_cell_183/MatMul/ReadVariableOp¢+while/lstm_cell_183/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
)while/lstm_cell_183/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_183_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0¼
while/lstm_cell_183/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_183/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
+while/lstm_cell_183/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_183_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0£
while/lstm_cell_183/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_183/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
while/lstm_cell_183/addAddV2$while/lstm_cell_183/MatMul:product:0&while/lstm_cell_183/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*while/lstm_cell_183/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_183_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0ª
while/lstm_cell_183/BiasAddBiasAddwhile/lstm_cell_183/add:z:02while/lstm_cell_183/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
#while/lstm_cell_183/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ò
while/lstm_cell_183/splitSplit,while/lstm_cell_183/split/split_dim:output:0$while/lstm_cell_183/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split|
while/lstm_cell_183/SigmoidSigmoid"while/lstm_cell_183/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
while/lstm_cell_183/Sigmoid_1Sigmoid"while/lstm_cell_183/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_183/mulMul!while/lstm_cell_183/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ v
while/lstm_cell_183/ReluRelu"while/lstm_cell_183/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_183/mul_1Mulwhile/lstm_cell_183/Sigmoid:y:0&while/lstm_cell_183/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_183/add_1AddV2while/lstm_cell_183/mul:z:0while/lstm_cell_183/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
while/lstm_cell_183/Sigmoid_2Sigmoid"while/lstm_cell_183/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ s
while/lstm_cell_183/Relu_1Reluwhile/lstm_cell_183/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_183/mul_2Mul!while/lstm_cell_183/Sigmoid_2:y:0(while/lstm_cell_183/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Æ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_183/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_183/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
while/Identity_5Identitywhile/lstm_cell_183/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ó

while/NoOpNoOp+^while/lstm_cell_183/BiasAdd/ReadVariableOp*^while/lstm_cell_183/MatMul/ReadVariableOp,^while/lstm_cell_183/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_183_biasadd_readvariableop_resource5while_lstm_cell_183_biasadd_readvariableop_resource_0"n
4while_lstm_cell_183_matmul_1_readvariableop_resource6while_lstm_cell_183_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_183_matmul_readvariableop_resource4while_lstm_cell_183_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2X
*while/lstm_cell_183/BiasAdd/ReadVariableOp*while/lstm_cell_183/BiasAdd/ReadVariableOp2V
)while/lstm_cell_183/MatMul/ReadVariableOp)while/lstm_cell_183/MatMul/ReadVariableOp2Z
+while/lstm_cell_183/MatMul_1/ReadVariableOp+while/lstm_cell_183/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
K
¤
D__inference_lstm_45_layer_call_and_return_conditional_losses_3534547
inputs_0?
,lstm_cell_183_matmul_readvariableop_resource:	A
.lstm_cell_183_matmul_1_readvariableop_resource:	 <
-lstm_cell_183_biasadd_readvariableop_resource:	
identity¢$lstm_cell_183/BiasAdd/ReadVariableOp¢#lstm_cell_183/MatMul/ReadVariableOp¢%lstm_cell_183/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
#lstm_cell_183/MatMul/ReadVariableOpReadVariableOp,lstm_cell_183_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_183/MatMulMatMulstrided_slice_2:output:0+lstm_cell_183/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%lstm_cell_183/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_183_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_183/MatMul_1MatMulzeros:output:0-lstm_cell_183/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_183/addAddV2lstm_cell_183/MatMul:product:0 lstm_cell_183/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_183/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_183_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_183/BiasAddBiasAddlstm_cell_183/add:z:0,lstm_cell_183/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
lstm_cell_183/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :à
lstm_cell_183/splitSplit&lstm_cell_183/split/split_dim:output:0lstm_cell_183/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitp
lstm_cell_183/SigmoidSigmoidlstm_cell_183/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
lstm_cell_183/Sigmoid_1Sigmoidlstm_cell_183/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y
lstm_cell_183/mulMullstm_cell_183/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
lstm_cell_183/ReluRelulstm_cell_183/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_183/mul_1Mullstm_cell_183/Sigmoid:y:0 lstm_cell_183/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
lstm_cell_183/add_1AddV2lstm_cell_183/mul:z:0lstm_cell_183/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
lstm_cell_183/Sigmoid_2Sigmoidlstm_cell_183/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ g
lstm_cell_183/Relu_1Relulstm_cell_183/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_183/mul_2Mullstm_cell_183/Sigmoid_2:y:0"lstm_cell_183/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_183_matmul_readvariableop_resource.lstm_cell_183_matmul_1_readvariableop_resource-lstm_cell_183_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3534463*
condR
while_cond_3534462*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ã
NoOpNoOp%^lstm_cell_183/BiasAdd/ReadVariableOp$^lstm_cell_183/MatMul/ReadVariableOp&^lstm_cell_183/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_183/BiasAdd/ReadVariableOp$lstm_cell_183/BiasAdd/ReadVariableOp2J
#lstm_cell_183/MatMul/ReadVariableOp#lstm_cell_183/MatMul/ReadVariableOp2N
%lstm_cell_183/MatMul_1/ReadVariableOp%lstm_cell_183/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ï
ø
/__inference_lstm_cell_183_layer_call_fn_3536287

inputs
states_0
states_1
unknown:	
	unknown_0:	 
	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_183_layer_call_and_return_conditional_losses_3531351o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/1
B
Ú

lstm_46_while_body_3534120,
(lstm_46_while_lstm_46_while_loop_counter2
.lstm_46_while_lstm_46_while_maximum_iterations
lstm_46_while_placeholder
lstm_46_while_placeholder_1
lstm_46_while_placeholder_2
lstm_46_while_placeholder_3+
'lstm_46_while_lstm_46_strided_slice_1_0g
clstm_46_while_tensorarrayv2read_tensorlistgetitem_lstm_46_tensorarrayunstack_tensorlistfromtensor_0O
<lstm_46_while_lstm_cell_184_matmul_readvariableop_resource_0:	 Q
>lstm_46_while_lstm_cell_184_matmul_1_readvariableop_resource_0:	 L
=lstm_46_while_lstm_cell_184_biasadd_readvariableop_resource_0:	
lstm_46_while_identity
lstm_46_while_identity_1
lstm_46_while_identity_2
lstm_46_while_identity_3
lstm_46_while_identity_4
lstm_46_while_identity_5)
%lstm_46_while_lstm_46_strided_slice_1e
alstm_46_while_tensorarrayv2read_tensorlistgetitem_lstm_46_tensorarrayunstack_tensorlistfromtensorM
:lstm_46_while_lstm_cell_184_matmul_readvariableop_resource:	 O
<lstm_46_while_lstm_cell_184_matmul_1_readvariableop_resource:	 J
;lstm_46_while_lstm_cell_184_biasadd_readvariableop_resource:	¢2lstm_46/while/lstm_cell_184/BiasAdd/ReadVariableOp¢1lstm_46/while/lstm_cell_184/MatMul/ReadVariableOp¢3lstm_46/while/lstm_cell_184/MatMul_1/ReadVariableOp
?lstm_46/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Î
1lstm_46/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_46_while_tensorarrayv2read_tensorlistgetitem_lstm_46_tensorarrayunstack_tensorlistfromtensor_0lstm_46_while_placeholderHlstm_46/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0¯
1lstm_46/while/lstm_cell_184/MatMul/ReadVariableOpReadVariableOp<lstm_46_while_lstm_cell_184_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype0Ô
"lstm_46/while/lstm_cell_184/MatMulMatMul8lstm_46/while/TensorArrayV2Read/TensorListGetItem:item:09lstm_46/while/lstm_cell_184/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
3lstm_46/while/lstm_cell_184/MatMul_1/ReadVariableOpReadVariableOp>lstm_46_while_lstm_cell_184_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0»
$lstm_46/while/lstm_cell_184/MatMul_1MatMullstm_46_while_placeholder_2;lstm_46/while/lstm_cell_184/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
lstm_46/while/lstm_cell_184/addAddV2,lstm_46/while/lstm_cell_184/MatMul:product:0.lstm_46/while/lstm_cell_184/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
2lstm_46/while/lstm_cell_184/BiasAdd/ReadVariableOpReadVariableOp=lstm_46_while_lstm_cell_184_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0Â
#lstm_46/while/lstm_cell_184/BiasAddBiasAdd#lstm_46/while/lstm_cell_184/add:z:0:lstm_46/while/lstm_cell_184/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
+lstm_46/while/lstm_cell_184/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
!lstm_46/while/lstm_cell_184/splitSplit4lstm_46/while/lstm_cell_184/split/split_dim:output:0,lstm_46/while/lstm_cell_184/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split
#lstm_46/while/lstm_cell_184/SigmoidSigmoid*lstm_46/while/lstm_cell_184/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
%lstm_46/while/lstm_cell_184/Sigmoid_1Sigmoid*lstm_46/while/lstm_cell_184/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
lstm_46/while/lstm_cell_184/mulMul)lstm_46/while/lstm_cell_184/Sigmoid_1:y:0lstm_46_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 lstm_46/while/lstm_cell_184/ReluRelu*lstm_46/while/lstm_cell_184/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ³
!lstm_46/while/lstm_cell_184/mul_1Mul'lstm_46/while/lstm_cell_184/Sigmoid:y:0.lstm_46/while/lstm_cell_184/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¨
!lstm_46/while/lstm_cell_184/add_1AddV2#lstm_46/while/lstm_cell_184/mul:z:0%lstm_46/while/lstm_cell_184/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
%lstm_46/while/lstm_cell_184/Sigmoid_2Sigmoid*lstm_46/while/lstm_cell_184/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"lstm_46/while/lstm_cell_184/Relu_1Relu%lstm_46/while/lstm_cell_184/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ·
!lstm_46/while/lstm_cell_184/mul_2Mul)lstm_46/while/lstm_cell_184/Sigmoid_2:y:00lstm_46/while/lstm_cell_184/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ æ
2lstm_46/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_46_while_placeholder_1lstm_46_while_placeholder%lstm_46/while/lstm_cell_184/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒU
lstm_46/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_46/while/addAddV2lstm_46_while_placeholderlstm_46/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_46/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_46/while/add_1AddV2(lstm_46_while_lstm_46_while_loop_counterlstm_46/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_46/while/IdentityIdentitylstm_46/while/add_1:z:0^lstm_46/while/NoOp*
T0*
_output_shapes
: 
lstm_46/while/Identity_1Identity.lstm_46_while_lstm_46_while_maximum_iterations^lstm_46/while/NoOp*
T0*
_output_shapes
: q
lstm_46/while/Identity_2Identitylstm_46/while/add:z:0^lstm_46/while/NoOp*
T0*
_output_shapes
: 
lstm_46/while/Identity_3IdentityBlstm_46/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_46/while/NoOp*
T0*
_output_shapes
: 
lstm_46/while/Identity_4Identity%lstm_46/while/lstm_cell_184/mul_2:z:0^lstm_46/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_46/while/Identity_5Identity%lstm_46/while/lstm_cell_184/add_1:z:0^lstm_46/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ó
lstm_46/while/NoOpNoOp3^lstm_46/while/lstm_cell_184/BiasAdd/ReadVariableOp2^lstm_46/while/lstm_cell_184/MatMul/ReadVariableOp4^lstm_46/while/lstm_cell_184/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_46_while_identitylstm_46/while/Identity:output:0"=
lstm_46_while_identity_1!lstm_46/while/Identity_1:output:0"=
lstm_46_while_identity_2!lstm_46/while/Identity_2:output:0"=
lstm_46_while_identity_3!lstm_46/while/Identity_3:output:0"=
lstm_46_while_identity_4!lstm_46/while/Identity_4:output:0"=
lstm_46_while_identity_5!lstm_46/while/Identity_5:output:0"P
%lstm_46_while_lstm_46_strided_slice_1'lstm_46_while_lstm_46_strided_slice_1_0"|
;lstm_46_while_lstm_cell_184_biasadd_readvariableop_resource=lstm_46_while_lstm_cell_184_biasadd_readvariableop_resource_0"~
<lstm_46_while_lstm_cell_184_matmul_1_readvariableop_resource>lstm_46_while_lstm_cell_184_matmul_1_readvariableop_resource_0"z
:lstm_46_while_lstm_cell_184_matmul_readvariableop_resource<lstm_46_while_lstm_cell_184_matmul_readvariableop_resource_0"È
alstm_46_while_tensorarrayv2read_tensorlistgetitem_lstm_46_tensorarrayunstack_tensorlistfromtensorclstm_46_while_tensorarrayv2read_tensorlistgetitem_lstm_46_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2h
2lstm_46/while/lstm_cell_184/BiasAdd/ReadVariableOp2lstm_46/while/lstm_cell_184/BiasAdd/ReadVariableOp2f
1lstm_46/while/lstm_cell_184/MatMul/ReadVariableOp1lstm_46/while/lstm_cell_184/MatMul/ReadVariableOp2j
3lstm_46/while/lstm_cell_184/MatMul_1/ReadVariableOp3lstm_46/while/lstm_cell_184/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
«
¸
)__inference_lstm_46_layer_call_fn_3534998
inputs_0
unknown:	 
	unknown_0:	 
	unknown_1:	
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_46_layer_call_and_return_conditional_losses_3531829|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0
L
¤
D__inference_lstm_47_layer_call_and_return_conditional_losses_3535781
inputs_0?
,lstm_cell_185_matmul_readvariableop_resource:	 A
.lstm_cell_185_matmul_1_readvariableop_resource:	 <
-lstm_cell_185_biasadd_readvariableop_resource:	
identity¢$lstm_cell_185/BiasAdd/ReadVariableOp¢#lstm_cell_185/MatMul/ReadVariableOp¢%lstm_cell_185/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask
#lstm_cell_185/MatMul/ReadVariableOpReadVariableOp,lstm_cell_185_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_185/MatMulMatMulstrided_slice_2:output:0+lstm_cell_185/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%lstm_cell_185/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_185_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_185/MatMul_1MatMulzeros:output:0-lstm_cell_185/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_185/addAddV2lstm_cell_185/MatMul:product:0 lstm_cell_185/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_185/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_185_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_185/BiasAddBiasAddlstm_cell_185/add:z:0,lstm_cell_185/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
lstm_cell_185/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :à
lstm_cell_185/splitSplit&lstm_cell_185/split/split_dim:output:0lstm_cell_185/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitp
lstm_cell_185/SigmoidSigmoidlstm_cell_185/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
lstm_cell_185/Sigmoid_1Sigmoidlstm_cell_185/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y
lstm_cell_185/mulMullstm_cell_185/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
lstm_cell_185/ReluRelulstm_cell_185/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_185/mul_1Mullstm_cell_185/Sigmoid:y:0 lstm_cell_185/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
lstm_cell_185/add_1AddV2lstm_cell_185/mul:z:0lstm_cell_185/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
lstm_cell_185/Sigmoid_2Sigmoidlstm_cell_185/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ g
lstm_cell_185/Relu_1Relulstm_cell_185/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_185/mul_2Mullstm_cell_185/Sigmoid_2:y:0"lstm_cell_185/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_185_matmul_readvariableop_resource.lstm_cell_185_matmul_1_readvariableop_resource-lstm_cell_185_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3535696*
condR
while_cond_3535695*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ã
NoOpNoOp%^lstm_cell_185/BiasAdd/ReadVariableOp$^lstm_cell_185/MatMul/ReadVariableOp&^lstm_cell_185/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : 2L
$lstm_cell_185/BiasAdd/ReadVariableOp$lstm_cell_185/BiasAdd/ReadVariableOp2J
#lstm_cell_185/MatMul/ReadVariableOp#lstm_cell_185/MatMul/ReadVariableOp2N
%lstm_cell_185/MatMul_1/ReadVariableOp%lstm_cell_185/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0
Ø

b
F__inference_reshape_8_layer_call_and_return_conditional_losses_3536253

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:h
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®8

D__inference_lstm_46_layer_call_and_return_conditional_losses_3531638

inputs(
lstm_cell_184_3531556:	 (
lstm_cell_184_3531558:	 $
lstm_cell_184_3531560:	
identity¢%lstm_cell_184/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maskþ
%lstm_cell_184/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_184_3531556lstm_cell_184_3531558lstm_cell_184_3531560*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_184_layer_call_and_return_conditional_losses_3531555n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¿
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_184_3531556lstm_cell_184_3531558lstm_cell_184_3531560*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3531569*
condR
while_cond_3531568*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ v
NoOpNoOp&^lstm_cell_184/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : 2N
%lstm_cell_184/StatefulPartitionedCall%lstm_cell_184/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
«$
ñ
while_body_3532113
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_185_3532137_0:	 0
while_lstm_cell_185_3532139_0:	 ,
while_lstm_cell_185_3532141_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_185_3532137:	 .
while_lstm_cell_185_3532139:	 *
while_lstm_cell_185_3532141:	¢+while/lstm_cell_185/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0¼
+while/lstm_cell_185/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_185_3532137_0while_lstm_cell_185_3532139_0while_lstm_cell_185_3532141_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_185_layer_call_and_return_conditional_losses_3532053r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:04while/lstm_cell_185/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity4while/lstm_cell_185/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/Identity_5Identity4while/lstm_cell_185/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z

while/NoOpNoOp,^while/lstm_cell_185/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"<
while_lstm_cell_185_3532137while_lstm_cell_185_3532137_0"<
while_lstm_cell_185_3532139while_lstm_cell_185_3532139_0"<
while_lstm_cell_185_3532141while_lstm_cell_185_3532141_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2Z
+while/lstm_cell_185/StatefulPartitionedCall+while/lstm_cell_185/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
Ø

b
F__inference_reshape_8_layer_call_and_return_conditional_losses_3532680

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:h
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ï
ø
/__inference_lstm_cell_183_layer_call_fn_3536270

inputs
states_0
states_1
unknown:	
	unknown_0:	 
	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_183_layer_call_and_return_conditional_losses_3531205o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/1
áJ
¢
D__inference_lstm_45_layer_call_and_return_conditional_losses_3534976

inputs?
,lstm_cell_183_matmul_readvariableop_resource:	A
.lstm_cell_183_matmul_1_readvariableop_resource:	 <
-lstm_cell_183_biasadd_readvariableop_resource:	
identity¢$lstm_cell_183/BiasAdd/ReadVariableOp¢#lstm_cell_183/MatMul/ReadVariableOp¢%lstm_cell_183/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
#lstm_cell_183/MatMul/ReadVariableOpReadVariableOp,lstm_cell_183_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_183/MatMulMatMulstrided_slice_2:output:0+lstm_cell_183/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%lstm_cell_183/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_183_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_183/MatMul_1MatMulzeros:output:0-lstm_cell_183/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_183/addAddV2lstm_cell_183/MatMul:product:0 lstm_cell_183/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_183/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_183_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_183/BiasAddBiasAddlstm_cell_183/add:z:0,lstm_cell_183/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
lstm_cell_183/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :à
lstm_cell_183/splitSplit&lstm_cell_183/split/split_dim:output:0lstm_cell_183/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitp
lstm_cell_183/SigmoidSigmoidlstm_cell_183/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
lstm_cell_183/Sigmoid_1Sigmoidlstm_cell_183/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y
lstm_cell_183/mulMullstm_cell_183/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
lstm_cell_183/ReluRelulstm_cell_183/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_183/mul_1Mullstm_cell_183/Sigmoid:y:0 lstm_cell_183/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
lstm_cell_183/add_1AddV2lstm_cell_183/mul:z:0lstm_cell_183/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
lstm_cell_183/Sigmoid_2Sigmoidlstm_cell_183/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ g
lstm_cell_183/Relu_1Relulstm_cell_183/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_183/mul_2Mullstm_cell_183/Sigmoid_2:y:0"lstm_cell_183/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_183_matmul_readvariableop_resource.lstm_cell_183_matmul_1_readvariableop_resource-lstm_cell_183_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3534892*
condR
while_cond_3534891*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 Ã
NoOpNoOp%^lstm_cell_183/BiasAdd/ReadVariableOp$^lstm_cell_183/MatMul/ReadVariableOp&^lstm_cell_183/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : : 2L
$lstm_cell_183/BiasAdd/ReadVariableOp$lstm_cell_183/BiasAdd/ReadVariableOp2J
#lstm_cell_183/MatMul/ReadVariableOp#lstm_cell_183/MatMul/ReadVariableOp2N
%lstm_cell_183/MatMul_1/ReadVariableOp%lstm_cell_183/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
9

D__inference_lstm_47_layer_call_and_return_conditional_losses_3532183

inputs(
lstm_cell_185_3532099:	 (
lstm_cell_185_3532101:	 $
lstm_cell_185_3532103:	
identity¢%lstm_cell_185/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maskþ
%lstm_cell_185/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_185_3532099lstm_cell_185_3532101lstm_cell_185_3532103*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_185_layer_call_and_return_conditional_losses_3532053n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¿
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_185_3532099lstm_cell_185_3532101lstm_cell_185_3532103*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3532113*
condR
while_cond_3532112*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ v
NoOpNoOp&^lstm_cell_185/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : 2N
%lstm_cell_185/StatefulPartitionedCall%lstm_cell_185/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
®C
Ú

lstm_47_while_body_3534260,
(lstm_47_while_lstm_47_while_loop_counter2
.lstm_47_while_lstm_47_while_maximum_iterations
lstm_47_while_placeholder
lstm_47_while_placeholder_1
lstm_47_while_placeholder_2
lstm_47_while_placeholder_3+
'lstm_47_while_lstm_47_strided_slice_1_0g
clstm_47_while_tensorarrayv2read_tensorlistgetitem_lstm_47_tensorarrayunstack_tensorlistfromtensor_0O
<lstm_47_while_lstm_cell_185_matmul_readvariableop_resource_0:	 Q
>lstm_47_while_lstm_cell_185_matmul_1_readvariableop_resource_0:	 L
=lstm_47_while_lstm_cell_185_biasadd_readvariableop_resource_0:	
lstm_47_while_identity
lstm_47_while_identity_1
lstm_47_while_identity_2
lstm_47_while_identity_3
lstm_47_while_identity_4
lstm_47_while_identity_5)
%lstm_47_while_lstm_47_strided_slice_1e
alstm_47_while_tensorarrayv2read_tensorlistgetitem_lstm_47_tensorarrayunstack_tensorlistfromtensorM
:lstm_47_while_lstm_cell_185_matmul_readvariableop_resource:	 O
<lstm_47_while_lstm_cell_185_matmul_1_readvariableop_resource:	 J
;lstm_47_while_lstm_cell_185_biasadd_readvariableop_resource:	¢2lstm_47/while/lstm_cell_185/BiasAdd/ReadVariableOp¢1lstm_47/while/lstm_cell_185/MatMul/ReadVariableOp¢3lstm_47/while/lstm_cell_185/MatMul_1/ReadVariableOp
?lstm_47/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Î
1lstm_47/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_47_while_tensorarrayv2read_tensorlistgetitem_lstm_47_tensorarrayunstack_tensorlistfromtensor_0lstm_47_while_placeholderHlstm_47/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0¯
1lstm_47/while/lstm_cell_185/MatMul/ReadVariableOpReadVariableOp<lstm_47_while_lstm_cell_185_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype0Ô
"lstm_47/while/lstm_cell_185/MatMulMatMul8lstm_47/while/TensorArrayV2Read/TensorListGetItem:item:09lstm_47/while/lstm_cell_185/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
3lstm_47/while/lstm_cell_185/MatMul_1/ReadVariableOpReadVariableOp>lstm_47_while_lstm_cell_185_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0»
$lstm_47/while/lstm_cell_185/MatMul_1MatMullstm_47_while_placeholder_2;lstm_47/while/lstm_cell_185/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
lstm_47/while/lstm_cell_185/addAddV2,lstm_47/while/lstm_cell_185/MatMul:product:0.lstm_47/while/lstm_cell_185/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
2lstm_47/while/lstm_cell_185/BiasAdd/ReadVariableOpReadVariableOp=lstm_47_while_lstm_cell_185_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0Â
#lstm_47/while/lstm_cell_185/BiasAddBiasAdd#lstm_47/while/lstm_cell_185/add:z:0:lstm_47/while/lstm_cell_185/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
+lstm_47/while/lstm_cell_185/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
!lstm_47/while/lstm_cell_185/splitSplit4lstm_47/while/lstm_cell_185/split/split_dim:output:0,lstm_47/while/lstm_cell_185/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split
#lstm_47/while/lstm_cell_185/SigmoidSigmoid*lstm_47/while/lstm_cell_185/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
%lstm_47/while/lstm_cell_185/Sigmoid_1Sigmoid*lstm_47/while/lstm_cell_185/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
lstm_47/while/lstm_cell_185/mulMul)lstm_47/while/lstm_cell_185/Sigmoid_1:y:0lstm_47_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 lstm_47/while/lstm_cell_185/ReluRelu*lstm_47/while/lstm_cell_185/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ³
!lstm_47/while/lstm_cell_185/mul_1Mul'lstm_47/while/lstm_cell_185/Sigmoid:y:0.lstm_47/while/lstm_cell_185/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¨
!lstm_47/while/lstm_cell_185/add_1AddV2#lstm_47/while/lstm_cell_185/mul:z:0%lstm_47/while/lstm_cell_185/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
%lstm_47/while/lstm_cell_185/Sigmoid_2Sigmoid*lstm_47/while/lstm_cell_185/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"lstm_47/while/lstm_cell_185/Relu_1Relu%lstm_47/while/lstm_cell_185/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ·
!lstm_47/while/lstm_cell_185/mul_2Mul)lstm_47/while/lstm_cell_185/Sigmoid_2:y:00lstm_47/while/lstm_cell_185/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
8lstm_47/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
2lstm_47/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_47_while_placeholder_1Alstm_47/while/TensorArrayV2Write/TensorListSetItem/index:output:0%lstm_47/while/lstm_cell_185/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒU
lstm_47/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_47/while/addAddV2lstm_47_while_placeholderlstm_47/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_47/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_47/while/add_1AddV2(lstm_47_while_lstm_47_while_loop_counterlstm_47/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_47/while/IdentityIdentitylstm_47/while/add_1:z:0^lstm_47/while/NoOp*
T0*
_output_shapes
: 
lstm_47/while/Identity_1Identity.lstm_47_while_lstm_47_while_maximum_iterations^lstm_47/while/NoOp*
T0*
_output_shapes
: q
lstm_47/while/Identity_2Identitylstm_47/while/add:z:0^lstm_47/while/NoOp*
T0*
_output_shapes
: 
lstm_47/while/Identity_3IdentityBlstm_47/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_47/while/NoOp*
T0*
_output_shapes
: 
lstm_47/while/Identity_4Identity%lstm_47/while/lstm_cell_185/mul_2:z:0^lstm_47/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_47/while/Identity_5Identity%lstm_47/while/lstm_cell_185/add_1:z:0^lstm_47/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ó
lstm_47/while/NoOpNoOp3^lstm_47/while/lstm_cell_185/BiasAdd/ReadVariableOp2^lstm_47/while/lstm_cell_185/MatMul/ReadVariableOp4^lstm_47/while/lstm_cell_185/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_47_while_identitylstm_47/while/Identity:output:0"=
lstm_47_while_identity_1!lstm_47/while/Identity_1:output:0"=
lstm_47_while_identity_2!lstm_47/while/Identity_2:output:0"=
lstm_47_while_identity_3!lstm_47/while/Identity_3:output:0"=
lstm_47_while_identity_4!lstm_47/while/Identity_4:output:0"=
lstm_47_while_identity_5!lstm_47/while/Identity_5:output:0"P
%lstm_47_while_lstm_47_strided_slice_1'lstm_47_while_lstm_47_strided_slice_1_0"|
;lstm_47_while_lstm_cell_185_biasadd_readvariableop_resource=lstm_47_while_lstm_cell_185_biasadd_readvariableop_resource_0"~
<lstm_47_while_lstm_cell_185_matmul_1_readvariableop_resource>lstm_47_while_lstm_cell_185_matmul_1_readvariableop_resource_0"z
:lstm_47_while_lstm_cell_185_matmul_readvariableop_resource<lstm_47_while_lstm_cell_185_matmul_readvariableop_resource_0"È
alstm_47_while_tensorarrayv2read_tensorlistgetitem_lstm_47_tensorarrayunstack_tensorlistfromtensorclstm_47_while_tensorarrayv2read_tensorlistgetitem_lstm_47_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2h
2lstm_47/while/lstm_cell_185/BiasAdd/ReadVariableOp2lstm_47/while/lstm_cell_185/BiasAdd/ReadVariableOp2f
1lstm_47/while/lstm_cell_185/MatMul/ReadVariableOp1lstm_47/while/lstm_cell_185/MatMul/ReadVariableOp2j
3lstm_47/while/lstm_cell_185/MatMul_1/ReadVariableOp3lstm_47/while/lstm_cell_185/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
Ñ8
Ú
while_body_3535222
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_184_matmul_readvariableop_resource_0:	 I
6while_lstm_cell_184_matmul_1_readvariableop_resource_0:	 D
5while_lstm_cell_184_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_184_matmul_readvariableop_resource:	 G
4while_lstm_cell_184_matmul_1_readvariableop_resource:	 B
3while_lstm_cell_184_biasadd_readvariableop_resource:	¢*while/lstm_cell_184/BiasAdd/ReadVariableOp¢)while/lstm_cell_184/MatMul/ReadVariableOp¢+while/lstm_cell_184/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0
)while/lstm_cell_184/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_184_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype0¼
while/lstm_cell_184/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_184/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
+while/lstm_cell_184/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_184_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0£
while/lstm_cell_184/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_184/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
while/lstm_cell_184/addAddV2$while/lstm_cell_184/MatMul:product:0&while/lstm_cell_184/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*while/lstm_cell_184/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_184_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0ª
while/lstm_cell_184/BiasAddBiasAddwhile/lstm_cell_184/add:z:02while/lstm_cell_184/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
#while/lstm_cell_184/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ò
while/lstm_cell_184/splitSplit,while/lstm_cell_184/split/split_dim:output:0$while/lstm_cell_184/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split|
while/lstm_cell_184/SigmoidSigmoid"while/lstm_cell_184/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
while/lstm_cell_184/Sigmoid_1Sigmoid"while/lstm_cell_184/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_184/mulMul!while/lstm_cell_184/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ v
while/lstm_cell_184/ReluRelu"while/lstm_cell_184/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_184/mul_1Mulwhile/lstm_cell_184/Sigmoid:y:0&while/lstm_cell_184/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_184/add_1AddV2while/lstm_cell_184/mul:z:0while/lstm_cell_184/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
while/lstm_cell_184/Sigmoid_2Sigmoid"while/lstm_cell_184/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ s
while/lstm_cell_184/Relu_1Reluwhile/lstm_cell_184/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_184/mul_2Mul!while/lstm_cell_184/Sigmoid_2:y:0(while/lstm_cell_184/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Æ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_184/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_184/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
while/Identity_5Identitywhile/lstm_cell_184/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ó

while/NoOpNoOp+^while/lstm_cell_184/BiasAdd/ReadVariableOp*^while/lstm_cell_184/MatMul/ReadVariableOp,^while/lstm_cell_184/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_184_biasadd_readvariableop_resource5while_lstm_cell_184_biasadd_readvariableop_resource_0"n
4while_lstm_cell_184_matmul_1_readvariableop_resource6while_lstm_cell_184_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_184_matmul_readvariableop_resource4while_lstm_cell_184_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2X
*while/lstm_cell_184/BiasAdd/ReadVariableOp*while/lstm_cell_184/BiasAdd/ReadVariableOp2V
)while/lstm_cell_184/MatMul/ReadVariableOp)while/lstm_cell_184/MatMul/ReadVariableOp2Z
+while/lstm_cell_184/MatMul_1/ReadVariableOp+while/lstm_cell_184/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
ï
ø
/__inference_lstm_cell_184_layer_call_fn_3536368

inputs
states_0
states_1
unknown:	 
	unknown_0:	 
	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_184_layer_call_and_return_conditional_losses_3531555o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/1
L
¤
D__inference_lstm_47_layer_call_and_return_conditional_losses_3535926
inputs_0?
,lstm_cell_185_matmul_readvariableop_resource:	 A
.lstm_cell_185_matmul_1_readvariableop_resource:	 <
-lstm_cell_185_biasadd_readvariableop_resource:	
identity¢$lstm_cell_185/BiasAdd/ReadVariableOp¢#lstm_cell_185/MatMul/ReadVariableOp¢%lstm_cell_185/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask
#lstm_cell_185/MatMul/ReadVariableOpReadVariableOp,lstm_cell_185_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_185/MatMulMatMulstrided_slice_2:output:0+lstm_cell_185/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%lstm_cell_185/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_185_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_185/MatMul_1MatMulzeros:output:0-lstm_cell_185/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_185/addAddV2lstm_cell_185/MatMul:product:0 lstm_cell_185/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_185/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_185_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_185/BiasAddBiasAddlstm_cell_185/add:z:0,lstm_cell_185/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
lstm_cell_185/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :à
lstm_cell_185/splitSplit&lstm_cell_185/split/split_dim:output:0lstm_cell_185/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitp
lstm_cell_185/SigmoidSigmoidlstm_cell_185/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
lstm_cell_185/Sigmoid_1Sigmoidlstm_cell_185/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y
lstm_cell_185/mulMullstm_cell_185/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
lstm_cell_185/ReluRelulstm_cell_185/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_185/mul_1Mullstm_cell_185/Sigmoid:y:0 lstm_cell_185/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
lstm_cell_185/add_1AddV2lstm_cell_185/mul:z:0lstm_cell_185/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
lstm_cell_185/Sigmoid_2Sigmoidlstm_cell_185/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ g
lstm_cell_185/Relu_1Relulstm_cell_185/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_185/mul_2Mullstm_cell_185/Sigmoid_2:y:0"lstm_cell_185/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_185_matmul_readvariableop_resource.lstm_cell_185_matmul_1_readvariableop_resource-lstm_cell_185_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3535841*
condR
while_cond_3535840*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ã
NoOpNoOp%^lstm_cell_185/BiasAdd/ReadVariableOp$^lstm_cell_185/MatMul/ReadVariableOp&^lstm_cell_185/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : 2L
$lstm_cell_185/BiasAdd/ReadVariableOp$lstm_cell_185/BiasAdd/ReadVariableOp2J
#lstm_cell_185/MatMul/ReadVariableOp#lstm_cell_185/MatMul/ReadVariableOp2N
%lstm_cell_185/MatMul_1/ReadVariableOp%lstm_cell_185/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0
áJ
¢
D__inference_lstm_46_layer_call_and_return_conditional_losses_3533047

inputs?
,lstm_cell_184_matmul_readvariableop_resource:	 A
.lstm_cell_184_matmul_1_readvariableop_resource:	 <
-lstm_cell_184_biasadd_readvariableop_resource:	
identity¢$lstm_cell_184/BiasAdd/ReadVariableOp¢#lstm_cell_184/MatMul/ReadVariableOp¢%lstm_cell_184/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿ D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask
#lstm_cell_184/MatMul/ReadVariableOpReadVariableOp,lstm_cell_184_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_184/MatMulMatMulstrided_slice_2:output:0+lstm_cell_184/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%lstm_cell_184/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_184_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_184/MatMul_1MatMulzeros:output:0-lstm_cell_184/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_184/addAddV2lstm_cell_184/MatMul:product:0 lstm_cell_184/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_184/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_184_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_184/BiasAddBiasAddlstm_cell_184/add:z:0,lstm_cell_184/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
lstm_cell_184/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :à
lstm_cell_184/splitSplit&lstm_cell_184/split/split_dim:output:0lstm_cell_184/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitp
lstm_cell_184/SigmoidSigmoidlstm_cell_184/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
lstm_cell_184/Sigmoid_1Sigmoidlstm_cell_184/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y
lstm_cell_184/mulMullstm_cell_184/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
lstm_cell_184/ReluRelulstm_cell_184/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_184/mul_1Mullstm_cell_184/Sigmoid:y:0 lstm_cell_184/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
lstm_cell_184/add_1AddV2lstm_cell_184/mul:z:0lstm_cell_184/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
lstm_cell_184/Sigmoid_2Sigmoidlstm_cell_184/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ g
lstm_cell_184/Relu_1Relulstm_cell_184/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_184/mul_2Mullstm_cell_184/Sigmoid_2:y:0"lstm_cell_184/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_184_matmul_readvariableop_resource.lstm_cell_184_matmul_1_readvariableop_resource-lstm_cell_184_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3532963*
condR
while_cond_3532962*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 Ã
NoOpNoOp%^lstm_cell_184/BiasAdd/ReadVariableOp$^lstm_cell_184/MatMul/ReadVariableOp&^lstm_cell_184/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
 : : : 2L
$lstm_cell_184/BiasAdd/ReadVariableOp$lstm_cell_184/BiasAdd/ReadVariableOp2J
#lstm_cell_184/MatMul/ReadVariableOp#lstm_cell_184/MatMul/ReadVariableOp2N
%lstm_cell_184/MatMul_1/ReadVariableOp%lstm_cell_184/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
 
_user_specified_nameinputs
¼
¢
"__inference__wrapped_model_3531138
input_1T
Arnn_model_21_lstm_45_lstm_cell_183_matmul_readvariableop_resource:	V
Crnn_model_21_lstm_45_lstm_cell_183_matmul_1_readvariableop_resource:	 Q
Brnn_model_21_lstm_45_lstm_cell_183_biasadd_readvariableop_resource:	T
Arnn_model_21_lstm_46_lstm_cell_184_matmul_readvariableop_resource:	 V
Crnn_model_21_lstm_46_lstm_cell_184_matmul_1_readvariableop_resource:	 Q
Brnn_model_21_lstm_46_lstm_cell_184_biasadd_readvariableop_resource:	T
Arnn_model_21_lstm_47_lstm_cell_185_matmul_readvariableop_resource:	 V
Crnn_model_21_lstm_47_lstm_cell_185_matmul_1_readvariableop_resource:	 Q
Brnn_model_21_lstm_47_lstm_cell_185_biasadd_readvariableop_resource:	F
4rnn_model_21_dense_21_matmul_readvariableop_resource: C
5rnn_model_21_dense_21_biasadd_readvariableop_resource:
identity¢,rnn_model_21/dense_21/BiasAdd/ReadVariableOp¢+rnn_model_21/dense_21/MatMul/ReadVariableOp¢9rnn_model_21/lstm_45/lstm_cell_183/BiasAdd/ReadVariableOp¢8rnn_model_21/lstm_45/lstm_cell_183/MatMul/ReadVariableOp¢:rnn_model_21/lstm_45/lstm_cell_183/MatMul_1/ReadVariableOp¢rnn_model_21/lstm_45/while¢9rnn_model_21/lstm_46/lstm_cell_184/BiasAdd/ReadVariableOp¢8rnn_model_21/lstm_46/lstm_cell_184/MatMul/ReadVariableOp¢:rnn_model_21/lstm_46/lstm_cell_184/MatMul_1/ReadVariableOp¢rnn_model_21/lstm_46/while¢9rnn_model_21/lstm_47/lstm_cell_185/BiasAdd/ReadVariableOp¢8rnn_model_21/lstm_47/lstm_cell_185/MatMul/ReadVariableOp¢:rnn_model_21/lstm_47/lstm_cell_185/MatMul_1/ReadVariableOp¢rnn_model_21/lstm_47/whileQ
rnn_model_21/lstm_45/ShapeShapeinput_1*
T0*
_output_shapes
:r
(rnn_model_21/lstm_45/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*rnn_model_21/lstm_45/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*rnn_model_21/lstm_45/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:º
"rnn_model_21/lstm_45/strided_sliceStridedSlice#rnn_model_21/lstm_45/Shape:output:01rnn_model_21/lstm_45/strided_slice/stack:output:03rnn_model_21/lstm_45/strided_slice/stack_1:output:03rnn_model_21/lstm_45/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#rnn_model_21/lstm_45/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : ²
!rnn_model_21/lstm_45/zeros/packedPack+rnn_model_21/lstm_45/strided_slice:output:0,rnn_model_21/lstm_45/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:e
 rnn_model_21/lstm_45/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    «
rnn_model_21/lstm_45/zerosFill*rnn_model_21/lstm_45/zeros/packed:output:0)rnn_model_21/lstm_45/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ g
%rnn_model_21/lstm_45/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : ¶
#rnn_model_21/lstm_45/zeros_1/packedPack+rnn_model_21/lstm_45/strided_slice:output:0.rnn_model_21/lstm_45/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:g
"rnn_model_21/lstm_45/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ±
rnn_model_21/lstm_45/zeros_1Fill,rnn_model_21/lstm_45/zeros_1/packed:output:0+rnn_model_21/lstm_45/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ x
#rnn_model_21/lstm_45/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
rnn_model_21/lstm_45/transpose	Transposeinput_1,rnn_model_21/lstm_45/transpose/perm:output:0*
T0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿn
rnn_model_21/lstm_45/Shape_1Shape"rnn_model_21/lstm_45/transpose:y:0*
T0*
_output_shapes
:t
*rnn_model_21/lstm_45/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,rnn_model_21/lstm_45/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,rnn_model_21/lstm_45/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
$rnn_model_21/lstm_45/strided_slice_1StridedSlice%rnn_model_21/lstm_45/Shape_1:output:03rnn_model_21/lstm_45/strided_slice_1/stack:output:05rnn_model_21/lstm_45/strided_slice_1/stack_1:output:05rnn_model_21/lstm_45/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
0rnn_model_21/lstm_45/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿó
"rnn_model_21/lstm_45/TensorArrayV2TensorListReserve9rnn_model_21/lstm_45/TensorArrayV2/element_shape:output:0-rnn_model_21/lstm_45/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Jrnn_model_21/lstm_45/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
<rnn_model_21/lstm_45/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"rnn_model_21/lstm_45/transpose:y:0Srnn_model_21/lstm_45/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒt
*rnn_model_21/lstm_45/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,rnn_model_21/lstm_45/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,rnn_model_21/lstm_45/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ò
$rnn_model_21/lstm_45/strided_slice_2StridedSlice"rnn_model_21/lstm_45/transpose:y:03rnn_model_21/lstm_45/strided_slice_2/stack:output:05rnn_model_21/lstm_45/strided_slice_2/stack_1:output:05rnn_model_21/lstm_45/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask»
8rnn_model_21/lstm_45/lstm_cell_183/MatMul/ReadVariableOpReadVariableOpArnn_model_21_lstm_45_lstm_cell_183_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0×
)rnn_model_21/lstm_45/lstm_cell_183/MatMulMatMul-rnn_model_21/lstm_45/strided_slice_2:output:0@rnn_model_21/lstm_45/lstm_cell_183/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
:rnn_model_21/lstm_45/lstm_cell_183/MatMul_1/ReadVariableOpReadVariableOpCrnn_model_21_lstm_45_lstm_cell_183_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0Ñ
+rnn_model_21/lstm_45/lstm_cell_183/MatMul_1MatMul#rnn_model_21/lstm_45/zeros:output:0Brnn_model_21/lstm_45/lstm_cell_183/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÎ
&rnn_model_21/lstm_45/lstm_cell_183/addAddV23rnn_model_21/lstm_45/lstm_cell_183/MatMul:product:05rnn_model_21/lstm_45/lstm_cell_183/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
9rnn_model_21/lstm_45/lstm_cell_183/BiasAdd/ReadVariableOpReadVariableOpBrnn_model_21_lstm_45_lstm_cell_183_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0×
*rnn_model_21/lstm_45/lstm_cell_183/BiasAddBiasAdd*rnn_model_21/lstm_45/lstm_cell_183/add:z:0Arnn_model_21/lstm_45/lstm_cell_183/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
2rnn_model_21/lstm_45/lstm_cell_183/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
(rnn_model_21/lstm_45/lstm_cell_183/splitSplit;rnn_model_21/lstm_45/lstm_cell_183/split/split_dim:output:03rnn_model_21/lstm_45/lstm_cell_183/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split
*rnn_model_21/lstm_45/lstm_cell_183/SigmoidSigmoid1rnn_model_21/lstm_45/lstm_cell_183/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
,rnn_model_21/lstm_45/lstm_cell_183/Sigmoid_1Sigmoid1rnn_model_21/lstm_45/lstm_cell_183/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¸
&rnn_model_21/lstm_45/lstm_cell_183/mulMul0rnn_model_21/lstm_45/lstm_cell_183/Sigmoid_1:y:0%rnn_model_21/lstm_45/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
'rnn_model_21/lstm_45/lstm_cell_183/ReluRelu1rnn_model_21/lstm_45/lstm_cell_183/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ È
(rnn_model_21/lstm_45/lstm_cell_183/mul_1Mul.rnn_model_21/lstm_45/lstm_cell_183/Sigmoid:y:05rnn_model_21/lstm_45/lstm_cell_183/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ½
(rnn_model_21/lstm_45/lstm_cell_183/add_1AddV2*rnn_model_21/lstm_45/lstm_cell_183/mul:z:0,rnn_model_21/lstm_45/lstm_cell_183/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
,rnn_model_21/lstm_45/lstm_cell_183/Sigmoid_2Sigmoid1rnn_model_21/lstm_45/lstm_cell_183/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
)rnn_model_21/lstm_45/lstm_cell_183/Relu_1Relu,rnn_model_21/lstm_45/lstm_cell_183/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ì
(rnn_model_21/lstm_45/lstm_cell_183/mul_2Mul0rnn_model_21/lstm_45/lstm_cell_183/Sigmoid_2:y:07rnn_model_21/lstm_45/lstm_cell_183/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
2rnn_model_21/lstm_45/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ÷
$rnn_model_21/lstm_45/TensorArrayV2_1TensorListReserve;rnn_model_21/lstm_45/TensorArrayV2_1/element_shape:output:0-rnn_model_21/lstm_45/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ[
rnn_model_21/lstm_45/timeConst*
_output_shapes
: *
dtype0*
value	B : x
-rnn_model_21/lstm_45/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿi
'rnn_model_21/lstm_45/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ­
rnn_model_21/lstm_45/whileWhile0rnn_model_21/lstm_45/while/loop_counter:output:06rnn_model_21/lstm_45/while/maximum_iterations:output:0"rnn_model_21/lstm_45/time:output:0-rnn_model_21/lstm_45/TensorArrayV2_1:handle:0#rnn_model_21/lstm_45/zeros:output:0%rnn_model_21/lstm_45/zeros_1:output:0-rnn_model_21/lstm_45/strided_slice_1:output:0Lrnn_model_21/lstm_45/TensorArrayUnstack/TensorListFromTensor:output_handle:0Arnn_model_21_lstm_45_lstm_cell_183_matmul_readvariableop_resourceCrnn_model_21_lstm_45_lstm_cell_183_matmul_1_readvariableop_resourceBrnn_model_21_lstm_45_lstm_cell_183_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *3
body+R)
'rnn_model_21_lstm_45_while_body_3530759*3
cond+R)
'rnn_model_21_lstm_45_while_cond_3530758*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
Ernn_model_21/lstm_45/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    
7rnn_model_21/lstm_45/TensorArrayV2Stack/TensorListStackTensorListStack#rnn_model_21/lstm_45/while:output:3Nrnn_model_21/lstm_45/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿ *
element_dtype0}
*rnn_model_21/lstm_45/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿv
,rnn_model_21/lstm_45/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,rnn_model_21/lstm_45/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ð
$rnn_model_21/lstm_45/strided_slice_3StridedSlice@rnn_model_21/lstm_45/TensorArrayV2Stack/TensorListStack:tensor:03rnn_model_21/lstm_45/strided_slice_3/stack:output:05rnn_model_21/lstm_45/strided_slice_3/stack_1:output:05rnn_model_21/lstm_45/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maskz
%rnn_model_21/lstm_45/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Õ
 rnn_model_21/lstm_45/transpose_1	Transpose@rnn_model_21/lstm_45/TensorArrayV2Stack/TensorListStack:tensor:0.rnn_model_21/lstm_45/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 p
rnn_model_21/lstm_45/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    n
rnn_model_21/lstm_46/ShapeShape$rnn_model_21/lstm_45/transpose_1:y:0*
T0*
_output_shapes
:r
(rnn_model_21/lstm_46/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*rnn_model_21/lstm_46/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*rnn_model_21/lstm_46/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:º
"rnn_model_21/lstm_46/strided_sliceStridedSlice#rnn_model_21/lstm_46/Shape:output:01rnn_model_21/lstm_46/strided_slice/stack:output:03rnn_model_21/lstm_46/strided_slice/stack_1:output:03rnn_model_21/lstm_46/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#rnn_model_21/lstm_46/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : ²
!rnn_model_21/lstm_46/zeros/packedPack+rnn_model_21/lstm_46/strided_slice:output:0,rnn_model_21/lstm_46/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:e
 rnn_model_21/lstm_46/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    «
rnn_model_21/lstm_46/zerosFill*rnn_model_21/lstm_46/zeros/packed:output:0)rnn_model_21/lstm_46/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ g
%rnn_model_21/lstm_46/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : ¶
#rnn_model_21/lstm_46/zeros_1/packedPack+rnn_model_21/lstm_46/strided_slice:output:0.rnn_model_21/lstm_46/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:g
"rnn_model_21/lstm_46/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ±
rnn_model_21/lstm_46/zeros_1Fill,rnn_model_21/lstm_46/zeros_1/packed:output:0+rnn_model_21/lstm_46/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ x
#rnn_model_21/lstm_46/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          µ
rnn_model_21/lstm_46/transpose	Transpose$rnn_model_21/lstm_45/transpose_1:y:0,rnn_model_21/lstm_46/transpose/perm:output:0*
T0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿ n
rnn_model_21/lstm_46/Shape_1Shape"rnn_model_21/lstm_46/transpose:y:0*
T0*
_output_shapes
:t
*rnn_model_21/lstm_46/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,rnn_model_21/lstm_46/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,rnn_model_21/lstm_46/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
$rnn_model_21/lstm_46/strided_slice_1StridedSlice%rnn_model_21/lstm_46/Shape_1:output:03rnn_model_21/lstm_46/strided_slice_1/stack:output:05rnn_model_21/lstm_46/strided_slice_1/stack_1:output:05rnn_model_21/lstm_46/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
0rnn_model_21/lstm_46/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿó
"rnn_model_21/lstm_46/TensorArrayV2TensorListReserve9rnn_model_21/lstm_46/TensorArrayV2/element_shape:output:0-rnn_model_21/lstm_46/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Jrnn_model_21/lstm_46/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    
<rnn_model_21/lstm_46/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"rnn_model_21/lstm_46/transpose:y:0Srnn_model_21/lstm_46/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒt
*rnn_model_21/lstm_46/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,rnn_model_21/lstm_46/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,rnn_model_21/lstm_46/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ò
$rnn_model_21/lstm_46/strided_slice_2StridedSlice"rnn_model_21/lstm_46/transpose:y:03rnn_model_21/lstm_46/strided_slice_2/stack:output:05rnn_model_21/lstm_46/strided_slice_2/stack_1:output:05rnn_model_21/lstm_46/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask»
8rnn_model_21/lstm_46/lstm_cell_184/MatMul/ReadVariableOpReadVariableOpArnn_model_21_lstm_46_lstm_cell_184_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0×
)rnn_model_21/lstm_46/lstm_cell_184/MatMulMatMul-rnn_model_21/lstm_46/strided_slice_2:output:0@rnn_model_21/lstm_46/lstm_cell_184/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
:rnn_model_21/lstm_46/lstm_cell_184/MatMul_1/ReadVariableOpReadVariableOpCrnn_model_21_lstm_46_lstm_cell_184_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0Ñ
+rnn_model_21/lstm_46/lstm_cell_184/MatMul_1MatMul#rnn_model_21/lstm_46/zeros:output:0Brnn_model_21/lstm_46/lstm_cell_184/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÎ
&rnn_model_21/lstm_46/lstm_cell_184/addAddV23rnn_model_21/lstm_46/lstm_cell_184/MatMul:product:05rnn_model_21/lstm_46/lstm_cell_184/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
9rnn_model_21/lstm_46/lstm_cell_184/BiasAdd/ReadVariableOpReadVariableOpBrnn_model_21_lstm_46_lstm_cell_184_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0×
*rnn_model_21/lstm_46/lstm_cell_184/BiasAddBiasAdd*rnn_model_21/lstm_46/lstm_cell_184/add:z:0Arnn_model_21/lstm_46/lstm_cell_184/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
2rnn_model_21/lstm_46/lstm_cell_184/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
(rnn_model_21/lstm_46/lstm_cell_184/splitSplit;rnn_model_21/lstm_46/lstm_cell_184/split/split_dim:output:03rnn_model_21/lstm_46/lstm_cell_184/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split
*rnn_model_21/lstm_46/lstm_cell_184/SigmoidSigmoid1rnn_model_21/lstm_46/lstm_cell_184/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
,rnn_model_21/lstm_46/lstm_cell_184/Sigmoid_1Sigmoid1rnn_model_21/lstm_46/lstm_cell_184/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¸
&rnn_model_21/lstm_46/lstm_cell_184/mulMul0rnn_model_21/lstm_46/lstm_cell_184/Sigmoid_1:y:0%rnn_model_21/lstm_46/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
'rnn_model_21/lstm_46/lstm_cell_184/ReluRelu1rnn_model_21/lstm_46/lstm_cell_184/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ È
(rnn_model_21/lstm_46/lstm_cell_184/mul_1Mul.rnn_model_21/lstm_46/lstm_cell_184/Sigmoid:y:05rnn_model_21/lstm_46/lstm_cell_184/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ½
(rnn_model_21/lstm_46/lstm_cell_184/add_1AddV2*rnn_model_21/lstm_46/lstm_cell_184/mul:z:0,rnn_model_21/lstm_46/lstm_cell_184/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
,rnn_model_21/lstm_46/lstm_cell_184/Sigmoid_2Sigmoid1rnn_model_21/lstm_46/lstm_cell_184/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
)rnn_model_21/lstm_46/lstm_cell_184/Relu_1Relu,rnn_model_21/lstm_46/lstm_cell_184/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ì
(rnn_model_21/lstm_46/lstm_cell_184/mul_2Mul0rnn_model_21/lstm_46/lstm_cell_184/Sigmoid_2:y:07rnn_model_21/lstm_46/lstm_cell_184/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
2rnn_model_21/lstm_46/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ÷
$rnn_model_21/lstm_46/TensorArrayV2_1TensorListReserve;rnn_model_21/lstm_46/TensorArrayV2_1/element_shape:output:0-rnn_model_21/lstm_46/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ[
rnn_model_21/lstm_46/timeConst*
_output_shapes
: *
dtype0*
value	B : x
-rnn_model_21/lstm_46/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿi
'rnn_model_21/lstm_46/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ­
rnn_model_21/lstm_46/whileWhile0rnn_model_21/lstm_46/while/loop_counter:output:06rnn_model_21/lstm_46/while/maximum_iterations:output:0"rnn_model_21/lstm_46/time:output:0-rnn_model_21/lstm_46/TensorArrayV2_1:handle:0#rnn_model_21/lstm_46/zeros:output:0%rnn_model_21/lstm_46/zeros_1:output:0-rnn_model_21/lstm_46/strided_slice_1:output:0Lrnn_model_21/lstm_46/TensorArrayUnstack/TensorListFromTensor:output_handle:0Arnn_model_21_lstm_46_lstm_cell_184_matmul_readvariableop_resourceCrnn_model_21_lstm_46_lstm_cell_184_matmul_1_readvariableop_resourceBrnn_model_21_lstm_46_lstm_cell_184_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *3
body+R)
'rnn_model_21_lstm_46_while_body_3530898*3
cond+R)
'rnn_model_21_lstm_46_while_cond_3530897*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
Ernn_model_21/lstm_46/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    
7rnn_model_21/lstm_46/TensorArrayV2Stack/TensorListStackTensorListStack#rnn_model_21/lstm_46/while:output:3Nrnn_model_21/lstm_46/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿ *
element_dtype0}
*rnn_model_21/lstm_46/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿv
,rnn_model_21/lstm_46/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,rnn_model_21/lstm_46/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ð
$rnn_model_21/lstm_46/strided_slice_3StridedSlice@rnn_model_21/lstm_46/TensorArrayV2Stack/TensorListStack:tensor:03rnn_model_21/lstm_46/strided_slice_3/stack:output:05rnn_model_21/lstm_46/strided_slice_3/stack_1:output:05rnn_model_21/lstm_46/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maskz
%rnn_model_21/lstm_46/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Õ
 rnn_model_21/lstm_46/transpose_1	Transpose@rnn_model_21/lstm_46/TensorArrayV2Stack/TensorListStack:tensor:0.rnn_model_21/lstm_46/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 p
rnn_model_21/lstm_46/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    n
rnn_model_21/lstm_47/ShapeShape$rnn_model_21/lstm_46/transpose_1:y:0*
T0*
_output_shapes
:r
(rnn_model_21/lstm_47/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*rnn_model_21/lstm_47/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*rnn_model_21/lstm_47/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:º
"rnn_model_21/lstm_47/strided_sliceStridedSlice#rnn_model_21/lstm_47/Shape:output:01rnn_model_21/lstm_47/strided_slice/stack:output:03rnn_model_21/lstm_47/strided_slice/stack_1:output:03rnn_model_21/lstm_47/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#rnn_model_21/lstm_47/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : ²
!rnn_model_21/lstm_47/zeros/packedPack+rnn_model_21/lstm_47/strided_slice:output:0,rnn_model_21/lstm_47/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:e
 rnn_model_21/lstm_47/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    «
rnn_model_21/lstm_47/zerosFill*rnn_model_21/lstm_47/zeros/packed:output:0)rnn_model_21/lstm_47/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ g
%rnn_model_21/lstm_47/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : ¶
#rnn_model_21/lstm_47/zeros_1/packedPack+rnn_model_21/lstm_47/strided_slice:output:0.rnn_model_21/lstm_47/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:g
"rnn_model_21/lstm_47/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ±
rnn_model_21/lstm_47/zeros_1Fill,rnn_model_21/lstm_47/zeros_1/packed:output:0+rnn_model_21/lstm_47/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ x
#rnn_model_21/lstm_47/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          µ
rnn_model_21/lstm_47/transpose	Transpose$rnn_model_21/lstm_46/transpose_1:y:0,rnn_model_21/lstm_47/transpose/perm:output:0*
T0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿ n
rnn_model_21/lstm_47/Shape_1Shape"rnn_model_21/lstm_47/transpose:y:0*
T0*
_output_shapes
:t
*rnn_model_21/lstm_47/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,rnn_model_21/lstm_47/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,rnn_model_21/lstm_47/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
$rnn_model_21/lstm_47/strided_slice_1StridedSlice%rnn_model_21/lstm_47/Shape_1:output:03rnn_model_21/lstm_47/strided_slice_1/stack:output:05rnn_model_21/lstm_47/strided_slice_1/stack_1:output:05rnn_model_21/lstm_47/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
0rnn_model_21/lstm_47/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿó
"rnn_model_21/lstm_47/TensorArrayV2TensorListReserve9rnn_model_21/lstm_47/TensorArrayV2/element_shape:output:0-rnn_model_21/lstm_47/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Jrnn_model_21/lstm_47/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    
<rnn_model_21/lstm_47/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"rnn_model_21/lstm_47/transpose:y:0Srnn_model_21/lstm_47/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒt
*rnn_model_21/lstm_47/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,rnn_model_21/lstm_47/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,rnn_model_21/lstm_47/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ò
$rnn_model_21/lstm_47/strided_slice_2StridedSlice"rnn_model_21/lstm_47/transpose:y:03rnn_model_21/lstm_47/strided_slice_2/stack:output:05rnn_model_21/lstm_47/strided_slice_2/stack_1:output:05rnn_model_21/lstm_47/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask»
8rnn_model_21/lstm_47/lstm_cell_185/MatMul/ReadVariableOpReadVariableOpArnn_model_21_lstm_47_lstm_cell_185_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0×
)rnn_model_21/lstm_47/lstm_cell_185/MatMulMatMul-rnn_model_21/lstm_47/strided_slice_2:output:0@rnn_model_21/lstm_47/lstm_cell_185/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
:rnn_model_21/lstm_47/lstm_cell_185/MatMul_1/ReadVariableOpReadVariableOpCrnn_model_21_lstm_47_lstm_cell_185_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0Ñ
+rnn_model_21/lstm_47/lstm_cell_185/MatMul_1MatMul#rnn_model_21/lstm_47/zeros:output:0Brnn_model_21/lstm_47/lstm_cell_185/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÎ
&rnn_model_21/lstm_47/lstm_cell_185/addAddV23rnn_model_21/lstm_47/lstm_cell_185/MatMul:product:05rnn_model_21/lstm_47/lstm_cell_185/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
9rnn_model_21/lstm_47/lstm_cell_185/BiasAdd/ReadVariableOpReadVariableOpBrnn_model_21_lstm_47_lstm_cell_185_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0×
*rnn_model_21/lstm_47/lstm_cell_185/BiasAddBiasAdd*rnn_model_21/lstm_47/lstm_cell_185/add:z:0Arnn_model_21/lstm_47/lstm_cell_185/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
2rnn_model_21/lstm_47/lstm_cell_185/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
(rnn_model_21/lstm_47/lstm_cell_185/splitSplit;rnn_model_21/lstm_47/lstm_cell_185/split/split_dim:output:03rnn_model_21/lstm_47/lstm_cell_185/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split
*rnn_model_21/lstm_47/lstm_cell_185/SigmoidSigmoid1rnn_model_21/lstm_47/lstm_cell_185/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
,rnn_model_21/lstm_47/lstm_cell_185/Sigmoid_1Sigmoid1rnn_model_21/lstm_47/lstm_cell_185/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¸
&rnn_model_21/lstm_47/lstm_cell_185/mulMul0rnn_model_21/lstm_47/lstm_cell_185/Sigmoid_1:y:0%rnn_model_21/lstm_47/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
'rnn_model_21/lstm_47/lstm_cell_185/ReluRelu1rnn_model_21/lstm_47/lstm_cell_185/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ È
(rnn_model_21/lstm_47/lstm_cell_185/mul_1Mul.rnn_model_21/lstm_47/lstm_cell_185/Sigmoid:y:05rnn_model_21/lstm_47/lstm_cell_185/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ½
(rnn_model_21/lstm_47/lstm_cell_185/add_1AddV2*rnn_model_21/lstm_47/lstm_cell_185/mul:z:0,rnn_model_21/lstm_47/lstm_cell_185/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
,rnn_model_21/lstm_47/lstm_cell_185/Sigmoid_2Sigmoid1rnn_model_21/lstm_47/lstm_cell_185/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
)rnn_model_21/lstm_47/lstm_cell_185/Relu_1Relu,rnn_model_21/lstm_47/lstm_cell_185/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ì
(rnn_model_21/lstm_47/lstm_cell_185/mul_2Mul0rnn_model_21/lstm_47/lstm_cell_185/Sigmoid_2:y:07rnn_model_21/lstm_47/lstm_cell_185/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
2rnn_model_21/lstm_47/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    s
1rnn_model_21/lstm_47/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
$rnn_model_21/lstm_47/TensorArrayV2_1TensorListReserve;rnn_model_21/lstm_47/TensorArrayV2_1/element_shape:output:0:rnn_model_21/lstm_47/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ[
rnn_model_21/lstm_47/timeConst*
_output_shapes
: *
dtype0*
value	B : x
-rnn_model_21/lstm_47/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿi
'rnn_model_21/lstm_47/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ­
rnn_model_21/lstm_47/whileWhile0rnn_model_21/lstm_47/while/loop_counter:output:06rnn_model_21/lstm_47/while/maximum_iterations:output:0"rnn_model_21/lstm_47/time:output:0-rnn_model_21/lstm_47/TensorArrayV2_1:handle:0#rnn_model_21/lstm_47/zeros:output:0%rnn_model_21/lstm_47/zeros_1:output:0-rnn_model_21/lstm_47/strided_slice_1:output:0Lrnn_model_21/lstm_47/TensorArrayUnstack/TensorListFromTensor:output_handle:0Arnn_model_21_lstm_47_lstm_cell_185_matmul_readvariableop_resourceCrnn_model_21_lstm_47_lstm_cell_185_matmul_1_readvariableop_resourceBrnn_model_21_lstm_47_lstm_cell_185_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *3
body+R)
'rnn_model_21_lstm_47_while_body_3531038*3
cond+R)
'rnn_model_21_lstm_47_while_cond_3531037*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
Ernn_model_21/lstm_47/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    
7rnn_model_21/lstm_47/TensorArrayV2Stack/TensorListStackTensorListStack#rnn_model_21/lstm_47/while:output:3Nrnn_model_21/lstm_47/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0*
num_elements}
*rnn_model_21/lstm_47/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿv
,rnn_model_21/lstm_47/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,rnn_model_21/lstm_47/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ð
$rnn_model_21/lstm_47/strided_slice_3StridedSlice@rnn_model_21/lstm_47/TensorArrayV2Stack/TensorListStack:tensor:03rnn_model_21/lstm_47/strided_slice_3/stack:output:05rnn_model_21/lstm_47/strided_slice_3/stack_1:output:05rnn_model_21/lstm_47/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maskz
%rnn_model_21/lstm_47/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Õ
 rnn_model_21/lstm_47/transpose_1	Transpose@rnn_model_21/lstm_47/TensorArrayV2Stack/TensorListStack:tensor:0.rnn_model_21/lstm_47/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
rnn_model_21/lstm_47/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *     
+rnn_model_21/dense_21/MatMul/ReadVariableOpReadVariableOp4rnn_model_21_dense_21_matmul_readvariableop_resource*
_output_shapes

: *
dtype0¼
rnn_model_21/dense_21/MatMulMatMul-rnn_model_21/lstm_47/strided_slice_3:output:03rnn_model_21/dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,rnn_model_21/dense_21/BiasAdd/ReadVariableOpReadVariableOp5rnn_model_21_dense_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¸
rnn_model_21/dense_21/BiasAddBiasAdd&rnn_model_21/dense_21/MatMul:product:04rnn_model_21/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
rnn_model_21/reshape_8/ShapeShape&rnn_model_21/dense_21/BiasAdd:output:0*
T0*
_output_shapes
:t
*rnn_model_21/reshape_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,rnn_model_21/reshape_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,rnn_model_21/reshape_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
$rnn_model_21/reshape_8/strided_sliceStridedSlice%rnn_model_21/reshape_8/Shape:output:03rnn_model_21/reshape_8/strided_slice/stack:output:05rnn_model_21/reshape_8/strided_slice/stack_1:output:05rnn_model_21/reshape_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
&rnn_model_21/reshape_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :h
&rnn_model_21/reshape_8/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :ë
$rnn_model_21/reshape_8/Reshape/shapePack-rnn_model_21/reshape_8/strided_slice:output:0/rnn_model_21/reshape_8/Reshape/shape/1:output:0/rnn_model_21/reshape_8/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:¶
rnn_model_21/reshape_8/ReshapeReshape&rnn_model_21/dense_21/BiasAdd:output:0-rnn_model_21/reshape_8/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
IdentityIdentity'rnn_model_21/reshape_8/Reshape:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp-^rnn_model_21/dense_21/BiasAdd/ReadVariableOp,^rnn_model_21/dense_21/MatMul/ReadVariableOp:^rnn_model_21/lstm_45/lstm_cell_183/BiasAdd/ReadVariableOp9^rnn_model_21/lstm_45/lstm_cell_183/MatMul/ReadVariableOp;^rnn_model_21/lstm_45/lstm_cell_183/MatMul_1/ReadVariableOp^rnn_model_21/lstm_45/while:^rnn_model_21/lstm_46/lstm_cell_184/BiasAdd/ReadVariableOp9^rnn_model_21/lstm_46/lstm_cell_184/MatMul/ReadVariableOp;^rnn_model_21/lstm_46/lstm_cell_184/MatMul_1/ReadVariableOp^rnn_model_21/lstm_46/while:^rnn_model_21/lstm_47/lstm_cell_185/BiasAdd/ReadVariableOp9^rnn_model_21/lstm_47/lstm_cell_185/MatMul/ReadVariableOp;^rnn_model_21/lstm_47/lstm_cell_185/MatMul_1/ReadVariableOp^rnn_model_21/lstm_47/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : 2\
,rnn_model_21/dense_21/BiasAdd/ReadVariableOp,rnn_model_21/dense_21/BiasAdd/ReadVariableOp2Z
+rnn_model_21/dense_21/MatMul/ReadVariableOp+rnn_model_21/dense_21/MatMul/ReadVariableOp2v
9rnn_model_21/lstm_45/lstm_cell_183/BiasAdd/ReadVariableOp9rnn_model_21/lstm_45/lstm_cell_183/BiasAdd/ReadVariableOp2t
8rnn_model_21/lstm_45/lstm_cell_183/MatMul/ReadVariableOp8rnn_model_21/lstm_45/lstm_cell_183/MatMul/ReadVariableOp2x
:rnn_model_21/lstm_45/lstm_cell_183/MatMul_1/ReadVariableOp:rnn_model_21/lstm_45/lstm_cell_183/MatMul_1/ReadVariableOp28
rnn_model_21/lstm_45/whilernn_model_21/lstm_45/while2v
9rnn_model_21/lstm_46/lstm_cell_184/BiasAdd/ReadVariableOp9rnn_model_21/lstm_46/lstm_cell_184/BiasAdd/ReadVariableOp2t
8rnn_model_21/lstm_46/lstm_cell_184/MatMul/ReadVariableOp8rnn_model_21/lstm_46/lstm_cell_184/MatMul/ReadVariableOp2x
:rnn_model_21/lstm_46/lstm_cell_184/MatMul_1/ReadVariableOp:rnn_model_21/lstm_46/lstm_cell_184/MatMul_1/ReadVariableOp28
rnn_model_21/lstm_46/whilernn_model_21/lstm_46/while2v
9rnn_model_21/lstm_47/lstm_cell_185/BiasAdd/ReadVariableOp9rnn_model_21/lstm_47/lstm_cell_185/BiasAdd/ReadVariableOp2t
8rnn_model_21/lstm_47/lstm_cell_185/MatMul/ReadVariableOp8rnn_model_21/lstm_47/lstm_cell_185/MatMul/ReadVariableOp2x
:rnn_model_21/lstm_47/lstm_cell_185/MatMul_1/ReadVariableOp:rnn_model_21/lstm_47/lstm_cell_185/MatMul_1/ReadVariableOp28
rnn_model_21/lstm_47/whilernn_model_21/lstm_47/while:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1


è
lstm_45_while_cond_3533542,
(lstm_45_while_lstm_45_while_loop_counter2
.lstm_45_while_lstm_45_while_maximum_iterations
lstm_45_while_placeholder
lstm_45_while_placeholder_1
lstm_45_while_placeholder_2
lstm_45_while_placeholder_3.
*lstm_45_while_less_lstm_45_strided_slice_1E
Alstm_45_while_lstm_45_while_cond_3533542___redundant_placeholder0E
Alstm_45_while_lstm_45_while_cond_3533542___redundant_placeholder1E
Alstm_45_while_lstm_45_while_cond_3533542___redundant_placeholder2E
Alstm_45_while_lstm_45_while_cond_3533542___redundant_placeholder3
lstm_45_while_identity

lstm_45/while/LessLesslstm_45_while_placeholder*lstm_45_while_less_lstm_45_strided_slice_1*
T0*
_output_shapes
: [
lstm_45/while/IdentityIdentitylstm_45/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_45_while_identitylstm_45/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
×

J__inference_lstm_cell_185_layer_call_and_return_conditional_losses_3531905

inputs

states
states_11
matmul_readvariableop_resource:	 3
 matmul_1_readvariableop_resource:	 .
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 *
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ N
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates

¶
)__inference_lstm_46_layer_call_fn_3535020

inputs
unknown:	 
	unknown_0:	 
	unknown_1:	
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_46_layer_call_and_return_conditional_losses_3533047s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
 : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
 
_user_specified_nameinputs
«$
ñ
while_body_3531920
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_185_3531944_0:	 0
while_lstm_cell_185_3531946_0:	 ,
while_lstm_cell_185_3531948_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_185_3531944:	 .
while_lstm_cell_185_3531946:	 *
while_lstm_cell_185_3531948:	¢+while/lstm_cell_185/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0¼
+while/lstm_cell_185/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_185_3531944_0while_lstm_cell_185_3531946_0while_lstm_cell_185_3531948_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_185_layer_call_and_return_conditional_losses_3531905r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:04while/lstm_cell_185/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity4while/lstm_cell_185/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/Identity_5Identity4while/lstm_cell_185/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z

while/NoOpNoOp,^while/lstm_cell_185/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"<
while_lstm_cell_185_3531944while_lstm_cell_185_3531944_0"<
while_lstm_cell_185_3531946while_lstm_cell_185_3531946_0"<
while_lstm_cell_185_3531948while_lstm_cell_185_3531948_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2Z
+while/lstm_cell_185/StatefulPartitionedCall+while/lstm_cell_185/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
ï
ø
/__inference_lstm_cell_185_layer_call_fn_3536466

inputs
states_0
states_1
unknown:	 
	unknown_0:	 
	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_185_layer_call_and_return_conditional_losses_3531905o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/1
í9
Ú
while_body_3535986
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_185_matmul_readvariableop_resource_0:	 I
6while_lstm_cell_185_matmul_1_readvariableop_resource_0:	 D
5while_lstm_cell_185_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_185_matmul_readvariableop_resource:	 G
4while_lstm_cell_185_matmul_1_readvariableop_resource:	 B
3while_lstm_cell_185_biasadd_readvariableop_resource:	¢*while/lstm_cell_185/BiasAdd/ReadVariableOp¢)while/lstm_cell_185/MatMul/ReadVariableOp¢+while/lstm_cell_185/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0
)while/lstm_cell_185/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_185_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype0¼
while/lstm_cell_185/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_185/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
+while/lstm_cell_185/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_185_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0£
while/lstm_cell_185/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_185/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
while/lstm_cell_185/addAddV2$while/lstm_cell_185/MatMul:product:0&while/lstm_cell_185/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*while/lstm_cell_185/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_185_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0ª
while/lstm_cell_185/BiasAddBiasAddwhile/lstm_cell_185/add:z:02while/lstm_cell_185/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
#while/lstm_cell_185/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ò
while/lstm_cell_185/splitSplit,while/lstm_cell_185/split/split_dim:output:0$while/lstm_cell_185/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split|
while/lstm_cell_185/SigmoidSigmoid"while/lstm_cell_185/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
while/lstm_cell_185/Sigmoid_1Sigmoid"while/lstm_cell_185/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_185/mulMul!while/lstm_cell_185/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ v
while/lstm_cell_185/ReluRelu"while/lstm_cell_185/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_185/mul_1Mulwhile/lstm_cell_185/Sigmoid:y:0&while/lstm_cell_185/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_185/add_1AddV2while/lstm_cell_185/mul:z:0while/lstm_cell_185/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
while/lstm_cell_185/Sigmoid_2Sigmoid"while/lstm_cell_185/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ s
while/lstm_cell_185/Relu_1Reluwhile/lstm_cell_185/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_185/mul_2Mul!while/lstm_cell_185/Sigmoid_2:y:0(while/lstm_cell_185/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : î
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_185/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_185/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
while/Identity_5Identitywhile/lstm_cell_185/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ó

while/NoOpNoOp+^while/lstm_cell_185/BiasAdd/ReadVariableOp*^while/lstm_cell_185/MatMul/ReadVariableOp,^while/lstm_cell_185/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_185_biasadd_readvariableop_resource5while_lstm_cell_185_biasadd_readvariableop_resource_0"n
4while_lstm_cell_185_matmul_1_readvariableop_resource6while_lstm_cell_185_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_185_matmul_readvariableop_resource4while_lstm_cell_185_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2X
*while/lstm_cell_185/BiasAdd/ReadVariableOp*while/lstm_cell_185/BiasAdd/ReadVariableOp2V
)while/lstm_cell_185/MatMul/ReadVariableOp)while/lstm_cell_185/MatMul/ReadVariableOp2Z
+while/lstm_cell_185/MatMul_1/ReadVariableOp+while/lstm_cell_185/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
«
¸
)__inference_lstm_45_layer_call_fn_3534371
inputs_0
unknown:	
	unknown_0:	 
	unknown_1:	
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_45_layer_call_and_return_conditional_losses_3531288|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0


è
lstm_47_while_cond_3533821,
(lstm_47_while_lstm_47_while_loop_counter2
.lstm_47_while_lstm_47_while_maximum_iterations
lstm_47_while_placeholder
lstm_47_while_placeholder_1
lstm_47_while_placeholder_2
lstm_47_while_placeholder_3.
*lstm_47_while_less_lstm_47_strided_slice_1E
Alstm_47_while_lstm_47_while_cond_3533821___redundant_placeholder0E
Alstm_47_while_lstm_47_while_cond_3533821___redundant_placeholder1E
Alstm_47_while_lstm_47_while_cond_3533821___redundant_placeholder2E
Alstm_47_while_lstm_47_while_cond_3533821___redundant_placeholder3
lstm_47_while_identity

lstm_47/while/LessLesslstm_47_while_placeholder*lstm_47_while_less_lstm_47_strided_slice_1*
T0*
_output_shapes
: [
lstm_47/while/IdentityIdentitylstm_47/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_47_while_identitylstm_47/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
í9
Ú
while_body_3536131
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_185_matmul_readvariableop_resource_0:	 I
6while_lstm_cell_185_matmul_1_readvariableop_resource_0:	 D
5while_lstm_cell_185_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_185_matmul_readvariableop_resource:	 G
4while_lstm_cell_185_matmul_1_readvariableop_resource:	 B
3while_lstm_cell_185_biasadd_readvariableop_resource:	¢*while/lstm_cell_185/BiasAdd/ReadVariableOp¢)while/lstm_cell_185/MatMul/ReadVariableOp¢+while/lstm_cell_185/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0
)while/lstm_cell_185/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_185_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype0¼
while/lstm_cell_185/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_185/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
+while/lstm_cell_185/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_185_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0£
while/lstm_cell_185/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_185/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
while/lstm_cell_185/addAddV2$while/lstm_cell_185/MatMul:product:0&while/lstm_cell_185/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*while/lstm_cell_185/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_185_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0ª
while/lstm_cell_185/BiasAddBiasAddwhile/lstm_cell_185/add:z:02while/lstm_cell_185/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
#while/lstm_cell_185/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ò
while/lstm_cell_185/splitSplit,while/lstm_cell_185/split/split_dim:output:0$while/lstm_cell_185/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split|
while/lstm_cell_185/SigmoidSigmoid"while/lstm_cell_185/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
while/lstm_cell_185/Sigmoid_1Sigmoid"while/lstm_cell_185/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_185/mulMul!while/lstm_cell_185/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ v
while/lstm_cell_185/ReluRelu"while/lstm_cell_185/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_185/mul_1Mulwhile/lstm_cell_185/Sigmoid:y:0&while/lstm_cell_185/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_185/add_1AddV2while/lstm_cell_185/mul:z:0while/lstm_cell_185/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
while/lstm_cell_185/Sigmoid_2Sigmoid"while/lstm_cell_185/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ s
while/lstm_cell_185/Relu_1Reluwhile/lstm_cell_185/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_185/mul_2Mul!while/lstm_cell_185/Sigmoid_2:y:0(while/lstm_cell_185/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : î
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_185/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_185/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
while/Identity_5Identitywhile/lstm_cell_185/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ó

while/NoOpNoOp+^while/lstm_cell_185/BiasAdd/ReadVariableOp*^while/lstm_cell_185/MatMul/ReadVariableOp,^while/lstm_cell_185/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_185_biasadd_readvariableop_resource5while_lstm_cell_185_biasadd_readvariableop_resource_0"n
4while_lstm_cell_185_matmul_1_readvariableop_resource6while_lstm_cell_185_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_185_matmul_readvariableop_resource4while_lstm_cell_185_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2X
*while/lstm_cell_185/BiasAdd/ReadVariableOp*while/lstm_cell_185/BiasAdd/ReadVariableOp2V
)while/lstm_cell_185/MatMul/ReadVariableOp)while/lstm_cell_185/MatMul/ReadVariableOp2Z
+while/lstm_cell_185/MatMul_1/ReadVariableOp+while/lstm_cell_185/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
Ñ8
Ú
while_body_3533128
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_183_matmul_readvariableop_resource_0:	I
6while_lstm_cell_183_matmul_1_readvariableop_resource_0:	 D
5while_lstm_cell_183_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_183_matmul_readvariableop_resource:	G
4while_lstm_cell_183_matmul_1_readvariableop_resource:	 B
3while_lstm_cell_183_biasadd_readvariableop_resource:	¢*while/lstm_cell_183/BiasAdd/ReadVariableOp¢)while/lstm_cell_183/MatMul/ReadVariableOp¢+while/lstm_cell_183/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
)while/lstm_cell_183/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_183_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0¼
while/lstm_cell_183/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_183/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
+while/lstm_cell_183/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_183_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0£
while/lstm_cell_183/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_183/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
while/lstm_cell_183/addAddV2$while/lstm_cell_183/MatMul:product:0&while/lstm_cell_183/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*while/lstm_cell_183/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_183_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0ª
while/lstm_cell_183/BiasAddBiasAddwhile/lstm_cell_183/add:z:02while/lstm_cell_183/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
#while/lstm_cell_183/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ò
while/lstm_cell_183/splitSplit,while/lstm_cell_183/split/split_dim:output:0$while/lstm_cell_183/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split|
while/lstm_cell_183/SigmoidSigmoid"while/lstm_cell_183/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
while/lstm_cell_183/Sigmoid_1Sigmoid"while/lstm_cell_183/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_183/mulMul!while/lstm_cell_183/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ v
while/lstm_cell_183/ReluRelu"while/lstm_cell_183/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_183/mul_1Mulwhile/lstm_cell_183/Sigmoid:y:0&while/lstm_cell_183/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_183/add_1AddV2while/lstm_cell_183/mul:z:0while/lstm_cell_183/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
while/lstm_cell_183/Sigmoid_2Sigmoid"while/lstm_cell_183/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ s
while/lstm_cell_183/Relu_1Reluwhile/lstm_cell_183/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_183/mul_2Mul!while/lstm_cell_183/Sigmoid_2:y:0(while/lstm_cell_183/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Æ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_183/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_183/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
while/Identity_5Identitywhile/lstm_cell_183/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ó

while/NoOpNoOp+^while/lstm_cell_183/BiasAdd/ReadVariableOp*^while/lstm_cell_183/MatMul/ReadVariableOp,^while/lstm_cell_183/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_183_biasadd_readvariableop_resource5while_lstm_cell_183_biasadd_readvariableop_resource_0"n
4while_lstm_cell_183_matmul_1_readvariableop_resource6while_lstm_cell_183_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_183_matmul_readvariableop_resource4while_lstm_cell_183_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2X
*while/lstm_cell_183/BiasAdd/ReadVariableOp*while/lstm_cell_183/BiasAdd/ReadVariableOp2V
)while/lstm_cell_183/MatMul/ReadVariableOp)while/lstm_cell_183/MatMul/ReadVariableOp2Z
+while/lstm_cell_183/MatMul_1/ReadVariableOp+while/lstm_cell_183/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
áJ
¢
D__inference_lstm_46_layer_call_and_return_conditional_losses_3535592

inputs?
,lstm_cell_184_matmul_readvariableop_resource:	 A
.lstm_cell_184_matmul_1_readvariableop_resource:	 <
-lstm_cell_184_biasadd_readvariableop_resource:	
identity¢$lstm_cell_184/BiasAdd/ReadVariableOp¢#lstm_cell_184/MatMul/ReadVariableOp¢%lstm_cell_184/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿ D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask
#lstm_cell_184/MatMul/ReadVariableOpReadVariableOp,lstm_cell_184_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_184/MatMulMatMulstrided_slice_2:output:0+lstm_cell_184/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%lstm_cell_184/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_184_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_184/MatMul_1MatMulzeros:output:0-lstm_cell_184/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_184/addAddV2lstm_cell_184/MatMul:product:0 lstm_cell_184/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_184/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_184_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_184/BiasAddBiasAddlstm_cell_184/add:z:0,lstm_cell_184/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
lstm_cell_184/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :à
lstm_cell_184/splitSplit&lstm_cell_184/split/split_dim:output:0lstm_cell_184/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitp
lstm_cell_184/SigmoidSigmoidlstm_cell_184/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
lstm_cell_184/Sigmoid_1Sigmoidlstm_cell_184/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y
lstm_cell_184/mulMullstm_cell_184/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
lstm_cell_184/ReluRelulstm_cell_184/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_184/mul_1Mullstm_cell_184/Sigmoid:y:0 lstm_cell_184/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
lstm_cell_184/add_1AddV2lstm_cell_184/mul:z:0lstm_cell_184/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
lstm_cell_184/Sigmoid_2Sigmoidlstm_cell_184/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ g
lstm_cell_184/Relu_1Relulstm_cell_184/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_184/mul_2Mullstm_cell_184/Sigmoid_2:y:0"lstm_cell_184/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_184_matmul_readvariableop_resource.lstm_cell_184_matmul_1_readvariableop_resource-lstm_cell_184_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3535508*
condR
while_cond_3535507*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 Ã
NoOpNoOp%^lstm_cell_184/BiasAdd/ReadVariableOp$^lstm_cell_184/MatMul/ReadVariableOp&^lstm_cell_184/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
 : : : 2L
$lstm_cell_184/BiasAdd/ReadVariableOp$lstm_cell_184/BiasAdd/ReadVariableOp2J
#lstm_cell_184/MatMul/ReadVariableOp#lstm_cell_184/MatMul/ReadVariableOp2N
%lstm_cell_184/MatMul_1/ReadVariableOp%lstm_cell_184/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
 
_user_specified_nameinputs
»
þ

I__inference_rnn_model_21_layer_call_and_return_conditional_losses_3533922
xG
4lstm_45_lstm_cell_183_matmul_readvariableop_resource:	I
6lstm_45_lstm_cell_183_matmul_1_readvariableop_resource:	 D
5lstm_45_lstm_cell_183_biasadd_readvariableop_resource:	G
4lstm_46_lstm_cell_184_matmul_readvariableop_resource:	 I
6lstm_46_lstm_cell_184_matmul_1_readvariableop_resource:	 D
5lstm_46_lstm_cell_184_biasadd_readvariableop_resource:	G
4lstm_47_lstm_cell_185_matmul_readvariableop_resource:	 I
6lstm_47_lstm_cell_185_matmul_1_readvariableop_resource:	 D
5lstm_47_lstm_cell_185_biasadd_readvariableop_resource:	9
'dense_21_matmul_readvariableop_resource: 6
(dense_21_biasadd_readvariableop_resource:
identity¢dense_21/BiasAdd/ReadVariableOp¢dense_21/MatMul/ReadVariableOp¢,lstm_45/lstm_cell_183/BiasAdd/ReadVariableOp¢+lstm_45/lstm_cell_183/MatMul/ReadVariableOp¢-lstm_45/lstm_cell_183/MatMul_1/ReadVariableOp¢lstm_45/while¢,lstm_46/lstm_cell_184/BiasAdd/ReadVariableOp¢+lstm_46/lstm_cell_184/MatMul/ReadVariableOp¢-lstm_46/lstm_cell_184/MatMul_1/ReadVariableOp¢lstm_46/while¢,lstm_47/lstm_cell_185/BiasAdd/ReadVariableOp¢+lstm_47/lstm_cell_185/MatMul/ReadVariableOp¢-lstm_47/lstm_cell_185/MatMul_1/ReadVariableOp¢lstm_47/while>
lstm_45/ShapeShapex*
T0*
_output_shapes
:e
lstm_45/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_45/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_45/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
lstm_45/strided_sliceStridedSlicelstm_45/Shape:output:0$lstm_45/strided_slice/stack:output:0&lstm_45/strided_slice/stack_1:output:0&lstm_45/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_45/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 
lstm_45/zeros/packedPacklstm_45/strided_slice:output:0lstm_45/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_45/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_45/zerosFilllstm_45/zeros/packed:output:0lstm_45/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z
lstm_45/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 
lstm_45/zeros_1/packedPacklstm_45/strided_slice:output:0!lstm_45/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_45/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_45/zeros_1Filllstm_45/zeros_1/packed:output:0lstm_45/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ k
lstm_45/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
lstm_45/transpose	Transposexlstm_45/transpose/perm:output:0*
T0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿT
lstm_45/Shape_1Shapelstm_45/transpose:y:0*
T0*
_output_shapes
:g
lstm_45/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_45/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_45/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_45/strided_slice_1StridedSlicelstm_45/Shape_1:output:0&lstm_45/strided_slice_1/stack:output:0(lstm_45/strided_slice_1/stack_1:output:0(lstm_45/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_45/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÌ
lstm_45/TensorArrayV2TensorListReserve,lstm_45/TensorArrayV2/element_shape:output:0 lstm_45/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
=lstm_45/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ø
/lstm_45/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_45/transpose:y:0Flstm_45/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒg
lstm_45/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_45/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_45/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_45/strided_slice_2StridedSlicelstm_45/transpose:y:0&lstm_45/strided_slice_2/stack:output:0(lstm_45/strided_slice_2/stack_1:output:0(lstm_45/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask¡
+lstm_45/lstm_cell_183/MatMul/ReadVariableOpReadVariableOp4lstm_45_lstm_cell_183_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0°
lstm_45/lstm_cell_183/MatMulMatMul lstm_45/strided_slice_2:output:03lstm_45/lstm_cell_183/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
-lstm_45/lstm_cell_183/MatMul_1/ReadVariableOpReadVariableOp6lstm_45_lstm_cell_183_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0ª
lstm_45/lstm_cell_183/MatMul_1MatMullstm_45/zeros:output:05lstm_45/lstm_cell_183/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
lstm_45/lstm_cell_183/addAddV2&lstm_45/lstm_cell_183/MatMul:product:0(lstm_45/lstm_cell_183/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,lstm_45/lstm_cell_183/BiasAdd/ReadVariableOpReadVariableOp5lstm_45_lstm_cell_183_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0°
lstm_45/lstm_cell_183/BiasAddBiasAddlstm_45/lstm_cell_183/add:z:04lstm_45/lstm_cell_183/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
%lstm_45/lstm_cell_183/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ø
lstm_45/lstm_cell_183/splitSplit.lstm_45/lstm_cell_183/split/split_dim:output:0&lstm_45/lstm_cell_183/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split
lstm_45/lstm_cell_183/SigmoidSigmoid$lstm_45/lstm_cell_183/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_45/lstm_cell_183/Sigmoid_1Sigmoid$lstm_45/lstm_cell_183/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_45/lstm_cell_183/mulMul#lstm_45/lstm_cell_183/Sigmoid_1:y:0lstm_45/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
lstm_45/lstm_cell_183/ReluRelu$lstm_45/lstm_cell_183/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¡
lstm_45/lstm_cell_183/mul_1Mul!lstm_45/lstm_cell_183/Sigmoid:y:0(lstm_45/lstm_cell_183/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_45/lstm_cell_183/add_1AddV2lstm_45/lstm_cell_183/mul:z:0lstm_45/lstm_cell_183/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_45/lstm_cell_183/Sigmoid_2Sigmoid$lstm_45/lstm_cell_183/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
lstm_45/lstm_cell_183/Relu_1Relulstm_45/lstm_cell_183/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¥
lstm_45/lstm_cell_183/mul_2Mul#lstm_45/lstm_cell_183/Sigmoid_2:y:0*lstm_45/lstm_cell_183/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ v
%lstm_45/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ð
lstm_45/TensorArrayV2_1TensorListReserve.lstm_45/TensorArrayV2_1/element_shape:output:0 lstm_45/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒN
lstm_45/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_45/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ\
lstm_45/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ÷
lstm_45/whileWhile#lstm_45/while/loop_counter:output:0)lstm_45/while/maximum_iterations:output:0lstm_45/time:output:0 lstm_45/TensorArrayV2_1:handle:0lstm_45/zeros:output:0lstm_45/zeros_1:output:0 lstm_45/strided_slice_1:output:0?lstm_45/TensorArrayUnstack/TensorListFromTensor:output_handle:04lstm_45_lstm_cell_183_matmul_readvariableop_resource6lstm_45_lstm_cell_183_matmul_1_readvariableop_resource5lstm_45_lstm_cell_183_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_45_while_body_3533543*&
condR
lstm_45_while_cond_3533542*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
8lstm_45/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ú
*lstm_45/TensorArrayV2Stack/TensorListStackTensorListStacklstm_45/while:output:3Alstm_45/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿ *
element_dtype0p
lstm_45/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿi
lstm_45/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_45/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¯
lstm_45/strided_slice_3StridedSlice3lstm_45/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_45/strided_slice_3/stack:output:0(lstm_45/strided_slice_3/stack_1:output:0(lstm_45/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maskm
lstm_45/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ®
lstm_45/transpose_1	Transpose3lstm_45/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_45/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 c
lstm_45/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    T
lstm_46/ShapeShapelstm_45/transpose_1:y:0*
T0*
_output_shapes
:e
lstm_46/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_46/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_46/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
lstm_46/strided_sliceStridedSlicelstm_46/Shape:output:0$lstm_46/strided_slice/stack:output:0&lstm_46/strided_slice/stack_1:output:0&lstm_46/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_46/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 
lstm_46/zeros/packedPacklstm_46/strided_slice:output:0lstm_46/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_46/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_46/zerosFilllstm_46/zeros/packed:output:0lstm_46/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z
lstm_46/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 
lstm_46/zeros_1/packedPacklstm_46/strided_slice:output:0!lstm_46/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_46/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_46/zeros_1Filllstm_46/zeros_1/packed:output:0lstm_46/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ k
lstm_46/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
lstm_46/transpose	Transposelstm_45/transpose_1:y:0lstm_46/transpose/perm:output:0*
T0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿ T
lstm_46/Shape_1Shapelstm_46/transpose:y:0*
T0*
_output_shapes
:g
lstm_46/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_46/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_46/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_46/strided_slice_1StridedSlicelstm_46/Shape_1:output:0&lstm_46/strided_slice_1/stack:output:0(lstm_46/strided_slice_1/stack_1:output:0(lstm_46/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_46/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÌ
lstm_46/TensorArrayV2TensorListReserve,lstm_46/TensorArrayV2/element_shape:output:0 lstm_46/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
=lstm_46/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ø
/lstm_46/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_46/transpose:y:0Flstm_46/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒg
lstm_46/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_46/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_46/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_46/strided_slice_2StridedSlicelstm_46/transpose:y:0&lstm_46/strided_slice_2/stack:output:0(lstm_46/strided_slice_2/stack_1:output:0(lstm_46/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask¡
+lstm_46/lstm_cell_184/MatMul/ReadVariableOpReadVariableOp4lstm_46_lstm_cell_184_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0°
lstm_46/lstm_cell_184/MatMulMatMul lstm_46/strided_slice_2:output:03lstm_46/lstm_cell_184/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
-lstm_46/lstm_cell_184/MatMul_1/ReadVariableOpReadVariableOp6lstm_46_lstm_cell_184_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0ª
lstm_46/lstm_cell_184/MatMul_1MatMullstm_46/zeros:output:05lstm_46/lstm_cell_184/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
lstm_46/lstm_cell_184/addAddV2&lstm_46/lstm_cell_184/MatMul:product:0(lstm_46/lstm_cell_184/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,lstm_46/lstm_cell_184/BiasAdd/ReadVariableOpReadVariableOp5lstm_46_lstm_cell_184_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0°
lstm_46/lstm_cell_184/BiasAddBiasAddlstm_46/lstm_cell_184/add:z:04lstm_46/lstm_cell_184/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
%lstm_46/lstm_cell_184/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ø
lstm_46/lstm_cell_184/splitSplit.lstm_46/lstm_cell_184/split/split_dim:output:0&lstm_46/lstm_cell_184/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split
lstm_46/lstm_cell_184/SigmoidSigmoid$lstm_46/lstm_cell_184/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_46/lstm_cell_184/Sigmoid_1Sigmoid$lstm_46/lstm_cell_184/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_46/lstm_cell_184/mulMul#lstm_46/lstm_cell_184/Sigmoid_1:y:0lstm_46/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
lstm_46/lstm_cell_184/ReluRelu$lstm_46/lstm_cell_184/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¡
lstm_46/lstm_cell_184/mul_1Mul!lstm_46/lstm_cell_184/Sigmoid:y:0(lstm_46/lstm_cell_184/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_46/lstm_cell_184/add_1AddV2lstm_46/lstm_cell_184/mul:z:0lstm_46/lstm_cell_184/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_46/lstm_cell_184/Sigmoid_2Sigmoid$lstm_46/lstm_cell_184/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
lstm_46/lstm_cell_184/Relu_1Relulstm_46/lstm_cell_184/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¥
lstm_46/lstm_cell_184/mul_2Mul#lstm_46/lstm_cell_184/Sigmoid_2:y:0*lstm_46/lstm_cell_184/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ v
%lstm_46/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ð
lstm_46/TensorArrayV2_1TensorListReserve.lstm_46/TensorArrayV2_1/element_shape:output:0 lstm_46/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒN
lstm_46/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_46/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ\
lstm_46/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ÷
lstm_46/whileWhile#lstm_46/while/loop_counter:output:0)lstm_46/while/maximum_iterations:output:0lstm_46/time:output:0 lstm_46/TensorArrayV2_1:handle:0lstm_46/zeros:output:0lstm_46/zeros_1:output:0 lstm_46/strided_slice_1:output:0?lstm_46/TensorArrayUnstack/TensorListFromTensor:output_handle:04lstm_46_lstm_cell_184_matmul_readvariableop_resource6lstm_46_lstm_cell_184_matmul_1_readvariableop_resource5lstm_46_lstm_cell_184_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_46_while_body_3533682*&
condR
lstm_46_while_cond_3533681*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
8lstm_46/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ú
*lstm_46/TensorArrayV2Stack/TensorListStackTensorListStacklstm_46/while:output:3Alstm_46/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿ *
element_dtype0p
lstm_46/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿi
lstm_46/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_46/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¯
lstm_46/strided_slice_3StridedSlice3lstm_46/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_46/strided_slice_3/stack:output:0(lstm_46/strided_slice_3/stack_1:output:0(lstm_46/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maskm
lstm_46/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ®
lstm_46/transpose_1	Transpose3lstm_46/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_46/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 c
lstm_46/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    T
lstm_47/ShapeShapelstm_46/transpose_1:y:0*
T0*
_output_shapes
:e
lstm_47/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_47/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_47/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
lstm_47/strided_sliceStridedSlicelstm_47/Shape:output:0$lstm_47/strided_slice/stack:output:0&lstm_47/strided_slice/stack_1:output:0&lstm_47/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_47/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 
lstm_47/zeros/packedPacklstm_47/strided_slice:output:0lstm_47/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_47/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_47/zerosFilllstm_47/zeros/packed:output:0lstm_47/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z
lstm_47/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 
lstm_47/zeros_1/packedPacklstm_47/strided_slice:output:0!lstm_47/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_47/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_47/zeros_1Filllstm_47/zeros_1/packed:output:0lstm_47/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ k
lstm_47/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
lstm_47/transpose	Transposelstm_46/transpose_1:y:0lstm_47/transpose/perm:output:0*
T0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿ T
lstm_47/Shape_1Shapelstm_47/transpose:y:0*
T0*
_output_shapes
:g
lstm_47/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_47/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_47/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_47/strided_slice_1StridedSlicelstm_47/Shape_1:output:0&lstm_47/strided_slice_1/stack:output:0(lstm_47/strided_slice_1/stack_1:output:0(lstm_47/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_47/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÌ
lstm_47/TensorArrayV2TensorListReserve,lstm_47/TensorArrayV2/element_shape:output:0 lstm_47/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
=lstm_47/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ø
/lstm_47/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_47/transpose:y:0Flstm_47/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒg
lstm_47/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_47/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_47/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_47/strided_slice_2StridedSlicelstm_47/transpose:y:0&lstm_47/strided_slice_2/stack:output:0(lstm_47/strided_slice_2/stack_1:output:0(lstm_47/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask¡
+lstm_47/lstm_cell_185/MatMul/ReadVariableOpReadVariableOp4lstm_47_lstm_cell_185_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0°
lstm_47/lstm_cell_185/MatMulMatMul lstm_47/strided_slice_2:output:03lstm_47/lstm_cell_185/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
-lstm_47/lstm_cell_185/MatMul_1/ReadVariableOpReadVariableOp6lstm_47_lstm_cell_185_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0ª
lstm_47/lstm_cell_185/MatMul_1MatMullstm_47/zeros:output:05lstm_47/lstm_cell_185/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
lstm_47/lstm_cell_185/addAddV2&lstm_47/lstm_cell_185/MatMul:product:0(lstm_47/lstm_cell_185/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,lstm_47/lstm_cell_185/BiasAdd/ReadVariableOpReadVariableOp5lstm_47_lstm_cell_185_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0°
lstm_47/lstm_cell_185/BiasAddBiasAddlstm_47/lstm_cell_185/add:z:04lstm_47/lstm_cell_185/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
%lstm_47/lstm_cell_185/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ø
lstm_47/lstm_cell_185/splitSplit.lstm_47/lstm_cell_185/split/split_dim:output:0&lstm_47/lstm_cell_185/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split
lstm_47/lstm_cell_185/SigmoidSigmoid$lstm_47/lstm_cell_185/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_47/lstm_cell_185/Sigmoid_1Sigmoid$lstm_47/lstm_cell_185/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_47/lstm_cell_185/mulMul#lstm_47/lstm_cell_185/Sigmoid_1:y:0lstm_47/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
lstm_47/lstm_cell_185/ReluRelu$lstm_47/lstm_cell_185/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¡
lstm_47/lstm_cell_185/mul_1Mul!lstm_47/lstm_cell_185/Sigmoid:y:0(lstm_47/lstm_cell_185/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_47/lstm_cell_185/add_1AddV2lstm_47/lstm_cell_185/mul:z:0lstm_47/lstm_cell_185/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_47/lstm_cell_185/Sigmoid_2Sigmoid$lstm_47/lstm_cell_185/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
lstm_47/lstm_cell_185/Relu_1Relulstm_47/lstm_cell_185/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¥
lstm_47/lstm_cell_185/mul_2Mul#lstm_47/lstm_cell_185/Sigmoid_2:y:0*lstm_47/lstm_cell_185/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ v
%lstm_47/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    f
$lstm_47/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_47/TensorArrayV2_1TensorListReserve.lstm_47/TensorArrayV2_1/element_shape:output:0-lstm_47/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒN
lstm_47/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_47/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ\
lstm_47/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ÷
lstm_47/whileWhile#lstm_47/while/loop_counter:output:0)lstm_47/while/maximum_iterations:output:0lstm_47/time:output:0 lstm_47/TensorArrayV2_1:handle:0lstm_47/zeros:output:0lstm_47/zeros_1:output:0 lstm_47/strided_slice_1:output:0?lstm_47/TensorArrayUnstack/TensorListFromTensor:output_handle:04lstm_47_lstm_cell_185_matmul_readvariableop_resource6lstm_47_lstm_cell_185_matmul_1_readvariableop_resource5lstm_47_lstm_cell_185_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_47_while_body_3533822*&
condR
lstm_47_while_cond_3533821*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
8lstm_47/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    î
*lstm_47/TensorArrayV2Stack/TensorListStackTensorListStacklstm_47/while:output:3Alstm_47/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0*
num_elementsp
lstm_47/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿi
lstm_47/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_47/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¯
lstm_47/strided_slice_3StridedSlice3lstm_47/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_47/strided_slice_3/stack:output:0(lstm_47/strided_slice_3/stack_1:output:0(lstm_47/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maskm
lstm_47/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ®
lstm_47/transpose_1	Transpose3lstm_47/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_47/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
lstm_47/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_21/MatMulMatMul lstm_47/strided_slice_3:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
reshape_8/ShapeShapedense_21/BiasAdd:output:0*
T0*
_output_shapes
:g
reshape_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_8/strided_sliceStridedSlicereshape_8/Shape:output:0&reshape_8/strided_slice/stack:output:0(reshape_8/strided_slice/stack_1:output:0(reshape_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_8/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :·
reshape_8/Reshape/shapePack reshape_8/strided_slice:output:0"reshape_8/Reshape/shape/1:output:0"reshape_8/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:
reshape_8/ReshapeReshapedense_21/BiasAdd:output:0 reshape_8/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
IdentityIdentityreshape_8/Reshape:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
NoOpNoOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp-^lstm_45/lstm_cell_183/BiasAdd/ReadVariableOp,^lstm_45/lstm_cell_183/MatMul/ReadVariableOp.^lstm_45/lstm_cell_183/MatMul_1/ReadVariableOp^lstm_45/while-^lstm_46/lstm_cell_184/BiasAdd/ReadVariableOp,^lstm_46/lstm_cell_184/MatMul/ReadVariableOp.^lstm_46/lstm_cell_184/MatMul_1/ReadVariableOp^lstm_46/while-^lstm_47/lstm_cell_185/BiasAdd/ReadVariableOp,^lstm_47/lstm_cell_185/MatMul/ReadVariableOp.^lstm_47/lstm_cell_185/MatMul_1/ReadVariableOp^lstm_47/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : 2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2\
,lstm_45/lstm_cell_183/BiasAdd/ReadVariableOp,lstm_45/lstm_cell_183/BiasAdd/ReadVariableOp2Z
+lstm_45/lstm_cell_183/MatMul/ReadVariableOp+lstm_45/lstm_cell_183/MatMul/ReadVariableOp2^
-lstm_45/lstm_cell_183/MatMul_1/ReadVariableOp-lstm_45/lstm_cell_183/MatMul_1/ReadVariableOp2
lstm_45/whilelstm_45/while2\
,lstm_46/lstm_cell_184/BiasAdd/ReadVariableOp,lstm_46/lstm_cell_184/BiasAdd/ReadVariableOp2Z
+lstm_46/lstm_cell_184/MatMul/ReadVariableOp+lstm_46/lstm_cell_184/MatMul/ReadVariableOp2^
-lstm_46/lstm_cell_184/MatMul_1/ReadVariableOp-lstm_46/lstm_cell_184/MatMul_1/ReadVariableOp2
lstm_46/whilelstm_46/while2\
,lstm_47/lstm_cell_185/BiasAdd/ReadVariableOp,lstm_47/lstm_cell_185/BiasAdd/ReadVariableOp2Z
+lstm_47/lstm_cell_185/MatMul/ReadVariableOp+lstm_47/lstm_cell_185/MatMul/ReadVariableOp2^
-lstm_47/lstm_cell_185/MatMul_1/ReadVariableOp-lstm_47/lstm_cell_185/MatMul_1/ReadVariableOp2
lstm_47/whilelstm_47/while:N J
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


_user_specified_namex
»
þ

I__inference_rnn_model_21_layer_call_and_return_conditional_losses_3534360
xG
4lstm_45_lstm_cell_183_matmul_readvariableop_resource:	I
6lstm_45_lstm_cell_183_matmul_1_readvariableop_resource:	 D
5lstm_45_lstm_cell_183_biasadd_readvariableop_resource:	G
4lstm_46_lstm_cell_184_matmul_readvariableop_resource:	 I
6lstm_46_lstm_cell_184_matmul_1_readvariableop_resource:	 D
5lstm_46_lstm_cell_184_biasadd_readvariableop_resource:	G
4lstm_47_lstm_cell_185_matmul_readvariableop_resource:	 I
6lstm_47_lstm_cell_185_matmul_1_readvariableop_resource:	 D
5lstm_47_lstm_cell_185_biasadd_readvariableop_resource:	9
'dense_21_matmul_readvariableop_resource: 6
(dense_21_biasadd_readvariableop_resource:
identity¢dense_21/BiasAdd/ReadVariableOp¢dense_21/MatMul/ReadVariableOp¢,lstm_45/lstm_cell_183/BiasAdd/ReadVariableOp¢+lstm_45/lstm_cell_183/MatMul/ReadVariableOp¢-lstm_45/lstm_cell_183/MatMul_1/ReadVariableOp¢lstm_45/while¢,lstm_46/lstm_cell_184/BiasAdd/ReadVariableOp¢+lstm_46/lstm_cell_184/MatMul/ReadVariableOp¢-lstm_46/lstm_cell_184/MatMul_1/ReadVariableOp¢lstm_46/while¢,lstm_47/lstm_cell_185/BiasAdd/ReadVariableOp¢+lstm_47/lstm_cell_185/MatMul/ReadVariableOp¢-lstm_47/lstm_cell_185/MatMul_1/ReadVariableOp¢lstm_47/while>
lstm_45/ShapeShapex*
T0*
_output_shapes
:e
lstm_45/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_45/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_45/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
lstm_45/strided_sliceStridedSlicelstm_45/Shape:output:0$lstm_45/strided_slice/stack:output:0&lstm_45/strided_slice/stack_1:output:0&lstm_45/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_45/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 
lstm_45/zeros/packedPacklstm_45/strided_slice:output:0lstm_45/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_45/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_45/zerosFilllstm_45/zeros/packed:output:0lstm_45/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z
lstm_45/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 
lstm_45/zeros_1/packedPacklstm_45/strided_slice:output:0!lstm_45/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_45/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_45/zeros_1Filllstm_45/zeros_1/packed:output:0lstm_45/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ k
lstm_45/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
lstm_45/transpose	Transposexlstm_45/transpose/perm:output:0*
T0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿT
lstm_45/Shape_1Shapelstm_45/transpose:y:0*
T0*
_output_shapes
:g
lstm_45/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_45/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_45/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_45/strided_slice_1StridedSlicelstm_45/Shape_1:output:0&lstm_45/strided_slice_1/stack:output:0(lstm_45/strided_slice_1/stack_1:output:0(lstm_45/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_45/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÌ
lstm_45/TensorArrayV2TensorListReserve,lstm_45/TensorArrayV2/element_shape:output:0 lstm_45/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
=lstm_45/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ø
/lstm_45/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_45/transpose:y:0Flstm_45/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒg
lstm_45/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_45/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_45/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_45/strided_slice_2StridedSlicelstm_45/transpose:y:0&lstm_45/strided_slice_2/stack:output:0(lstm_45/strided_slice_2/stack_1:output:0(lstm_45/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask¡
+lstm_45/lstm_cell_183/MatMul/ReadVariableOpReadVariableOp4lstm_45_lstm_cell_183_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0°
lstm_45/lstm_cell_183/MatMulMatMul lstm_45/strided_slice_2:output:03lstm_45/lstm_cell_183/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
-lstm_45/lstm_cell_183/MatMul_1/ReadVariableOpReadVariableOp6lstm_45_lstm_cell_183_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0ª
lstm_45/lstm_cell_183/MatMul_1MatMullstm_45/zeros:output:05lstm_45/lstm_cell_183/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
lstm_45/lstm_cell_183/addAddV2&lstm_45/lstm_cell_183/MatMul:product:0(lstm_45/lstm_cell_183/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,lstm_45/lstm_cell_183/BiasAdd/ReadVariableOpReadVariableOp5lstm_45_lstm_cell_183_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0°
lstm_45/lstm_cell_183/BiasAddBiasAddlstm_45/lstm_cell_183/add:z:04lstm_45/lstm_cell_183/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
%lstm_45/lstm_cell_183/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ø
lstm_45/lstm_cell_183/splitSplit.lstm_45/lstm_cell_183/split/split_dim:output:0&lstm_45/lstm_cell_183/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split
lstm_45/lstm_cell_183/SigmoidSigmoid$lstm_45/lstm_cell_183/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_45/lstm_cell_183/Sigmoid_1Sigmoid$lstm_45/lstm_cell_183/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_45/lstm_cell_183/mulMul#lstm_45/lstm_cell_183/Sigmoid_1:y:0lstm_45/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
lstm_45/lstm_cell_183/ReluRelu$lstm_45/lstm_cell_183/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¡
lstm_45/lstm_cell_183/mul_1Mul!lstm_45/lstm_cell_183/Sigmoid:y:0(lstm_45/lstm_cell_183/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_45/lstm_cell_183/add_1AddV2lstm_45/lstm_cell_183/mul:z:0lstm_45/lstm_cell_183/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_45/lstm_cell_183/Sigmoid_2Sigmoid$lstm_45/lstm_cell_183/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
lstm_45/lstm_cell_183/Relu_1Relulstm_45/lstm_cell_183/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¥
lstm_45/lstm_cell_183/mul_2Mul#lstm_45/lstm_cell_183/Sigmoid_2:y:0*lstm_45/lstm_cell_183/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ v
%lstm_45/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ð
lstm_45/TensorArrayV2_1TensorListReserve.lstm_45/TensorArrayV2_1/element_shape:output:0 lstm_45/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒN
lstm_45/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_45/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ\
lstm_45/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ÷
lstm_45/whileWhile#lstm_45/while/loop_counter:output:0)lstm_45/while/maximum_iterations:output:0lstm_45/time:output:0 lstm_45/TensorArrayV2_1:handle:0lstm_45/zeros:output:0lstm_45/zeros_1:output:0 lstm_45/strided_slice_1:output:0?lstm_45/TensorArrayUnstack/TensorListFromTensor:output_handle:04lstm_45_lstm_cell_183_matmul_readvariableop_resource6lstm_45_lstm_cell_183_matmul_1_readvariableop_resource5lstm_45_lstm_cell_183_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_45_while_body_3533981*&
condR
lstm_45_while_cond_3533980*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
8lstm_45/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ú
*lstm_45/TensorArrayV2Stack/TensorListStackTensorListStacklstm_45/while:output:3Alstm_45/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿ *
element_dtype0p
lstm_45/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿi
lstm_45/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_45/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¯
lstm_45/strided_slice_3StridedSlice3lstm_45/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_45/strided_slice_3/stack:output:0(lstm_45/strided_slice_3/stack_1:output:0(lstm_45/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maskm
lstm_45/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ®
lstm_45/transpose_1	Transpose3lstm_45/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_45/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 c
lstm_45/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    T
lstm_46/ShapeShapelstm_45/transpose_1:y:0*
T0*
_output_shapes
:e
lstm_46/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_46/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_46/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
lstm_46/strided_sliceStridedSlicelstm_46/Shape:output:0$lstm_46/strided_slice/stack:output:0&lstm_46/strided_slice/stack_1:output:0&lstm_46/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_46/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 
lstm_46/zeros/packedPacklstm_46/strided_slice:output:0lstm_46/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_46/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_46/zerosFilllstm_46/zeros/packed:output:0lstm_46/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z
lstm_46/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 
lstm_46/zeros_1/packedPacklstm_46/strided_slice:output:0!lstm_46/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_46/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_46/zeros_1Filllstm_46/zeros_1/packed:output:0lstm_46/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ k
lstm_46/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
lstm_46/transpose	Transposelstm_45/transpose_1:y:0lstm_46/transpose/perm:output:0*
T0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿ T
lstm_46/Shape_1Shapelstm_46/transpose:y:0*
T0*
_output_shapes
:g
lstm_46/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_46/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_46/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_46/strided_slice_1StridedSlicelstm_46/Shape_1:output:0&lstm_46/strided_slice_1/stack:output:0(lstm_46/strided_slice_1/stack_1:output:0(lstm_46/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_46/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÌ
lstm_46/TensorArrayV2TensorListReserve,lstm_46/TensorArrayV2/element_shape:output:0 lstm_46/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
=lstm_46/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ø
/lstm_46/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_46/transpose:y:0Flstm_46/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒg
lstm_46/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_46/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_46/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_46/strided_slice_2StridedSlicelstm_46/transpose:y:0&lstm_46/strided_slice_2/stack:output:0(lstm_46/strided_slice_2/stack_1:output:0(lstm_46/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask¡
+lstm_46/lstm_cell_184/MatMul/ReadVariableOpReadVariableOp4lstm_46_lstm_cell_184_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0°
lstm_46/lstm_cell_184/MatMulMatMul lstm_46/strided_slice_2:output:03lstm_46/lstm_cell_184/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
-lstm_46/lstm_cell_184/MatMul_1/ReadVariableOpReadVariableOp6lstm_46_lstm_cell_184_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0ª
lstm_46/lstm_cell_184/MatMul_1MatMullstm_46/zeros:output:05lstm_46/lstm_cell_184/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
lstm_46/lstm_cell_184/addAddV2&lstm_46/lstm_cell_184/MatMul:product:0(lstm_46/lstm_cell_184/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,lstm_46/lstm_cell_184/BiasAdd/ReadVariableOpReadVariableOp5lstm_46_lstm_cell_184_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0°
lstm_46/lstm_cell_184/BiasAddBiasAddlstm_46/lstm_cell_184/add:z:04lstm_46/lstm_cell_184/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
%lstm_46/lstm_cell_184/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ø
lstm_46/lstm_cell_184/splitSplit.lstm_46/lstm_cell_184/split/split_dim:output:0&lstm_46/lstm_cell_184/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split
lstm_46/lstm_cell_184/SigmoidSigmoid$lstm_46/lstm_cell_184/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_46/lstm_cell_184/Sigmoid_1Sigmoid$lstm_46/lstm_cell_184/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_46/lstm_cell_184/mulMul#lstm_46/lstm_cell_184/Sigmoid_1:y:0lstm_46/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
lstm_46/lstm_cell_184/ReluRelu$lstm_46/lstm_cell_184/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¡
lstm_46/lstm_cell_184/mul_1Mul!lstm_46/lstm_cell_184/Sigmoid:y:0(lstm_46/lstm_cell_184/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_46/lstm_cell_184/add_1AddV2lstm_46/lstm_cell_184/mul:z:0lstm_46/lstm_cell_184/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_46/lstm_cell_184/Sigmoid_2Sigmoid$lstm_46/lstm_cell_184/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
lstm_46/lstm_cell_184/Relu_1Relulstm_46/lstm_cell_184/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¥
lstm_46/lstm_cell_184/mul_2Mul#lstm_46/lstm_cell_184/Sigmoid_2:y:0*lstm_46/lstm_cell_184/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ v
%lstm_46/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ð
lstm_46/TensorArrayV2_1TensorListReserve.lstm_46/TensorArrayV2_1/element_shape:output:0 lstm_46/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒN
lstm_46/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_46/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ\
lstm_46/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ÷
lstm_46/whileWhile#lstm_46/while/loop_counter:output:0)lstm_46/while/maximum_iterations:output:0lstm_46/time:output:0 lstm_46/TensorArrayV2_1:handle:0lstm_46/zeros:output:0lstm_46/zeros_1:output:0 lstm_46/strided_slice_1:output:0?lstm_46/TensorArrayUnstack/TensorListFromTensor:output_handle:04lstm_46_lstm_cell_184_matmul_readvariableop_resource6lstm_46_lstm_cell_184_matmul_1_readvariableop_resource5lstm_46_lstm_cell_184_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_46_while_body_3534120*&
condR
lstm_46_while_cond_3534119*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
8lstm_46/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ú
*lstm_46/TensorArrayV2Stack/TensorListStackTensorListStacklstm_46/while:output:3Alstm_46/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿ *
element_dtype0p
lstm_46/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿi
lstm_46/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_46/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¯
lstm_46/strided_slice_3StridedSlice3lstm_46/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_46/strided_slice_3/stack:output:0(lstm_46/strided_slice_3/stack_1:output:0(lstm_46/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maskm
lstm_46/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ®
lstm_46/transpose_1	Transpose3lstm_46/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_46/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 c
lstm_46/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    T
lstm_47/ShapeShapelstm_46/transpose_1:y:0*
T0*
_output_shapes
:e
lstm_47/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_47/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_47/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
lstm_47/strided_sliceStridedSlicelstm_47/Shape:output:0$lstm_47/strided_slice/stack:output:0&lstm_47/strided_slice/stack_1:output:0&lstm_47/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_47/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 
lstm_47/zeros/packedPacklstm_47/strided_slice:output:0lstm_47/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_47/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_47/zerosFilllstm_47/zeros/packed:output:0lstm_47/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z
lstm_47/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 
lstm_47/zeros_1/packedPacklstm_47/strided_slice:output:0!lstm_47/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_47/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_47/zeros_1Filllstm_47/zeros_1/packed:output:0lstm_47/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ k
lstm_47/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
lstm_47/transpose	Transposelstm_46/transpose_1:y:0lstm_47/transpose/perm:output:0*
T0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿ T
lstm_47/Shape_1Shapelstm_47/transpose:y:0*
T0*
_output_shapes
:g
lstm_47/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_47/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_47/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_47/strided_slice_1StridedSlicelstm_47/Shape_1:output:0&lstm_47/strided_slice_1/stack:output:0(lstm_47/strided_slice_1/stack_1:output:0(lstm_47/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_47/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÌ
lstm_47/TensorArrayV2TensorListReserve,lstm_47/TensorArrayV2/element_shape:output:0 lstm_47/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
=lstm_47/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ø
/lstm_47/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_47/transpose:y:0Flstm_47/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒg
lstm_47/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_47/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_47/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_47/strided_slice_2StridedSlicelstm_47/transpose:y:0&lstm_47/strided_slice_2/stack:output:0(lstm_47/strided_slice_2/stack_1:output:0(lstm_47/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask¡
+lstm_47/lstm_cell_185/MatMul/ReadVariableOpReadVariableOp4lstm_47_lstm_cell_185_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0°
lstm_47/lstm_cell_185/MatMulMatMul lstm_47/strided_slice_2:output:03lstm_47/lstm_cell_185/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
-lstm_47/lstm_cell_185/MatMul_1/ReadVariableOpReadVariableOp6lstm_47_lstm_cell_185_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0ª
lstm_47/lstm_cell_185/MatMul_1MatMullstm_47/zeros:output:05lstm_47/lstm_cell_185/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
lstm_47/lstm_cell_185/addAddV2&lstm_47/lstm_cell_185/MatMul:product:0(lstm_47/lstm_cell_185/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,lstm_47/lstm_cell_185/BiasAdd/ReadVariableOpReadVariableOp5lstm_47_lstm_cell_185_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0°
lstm_47/lstm_cell_185/BiasAddBiasAddlstm_47/lstm_cell_185/add:z:04lstm_47/lstm_cell_185/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
%lstm_47/lstm_cell_185/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ø
lstm_47/lstm_cell_185/splitSplit.lstm_47/lstm_cell_185/split/split_dim:output:0&lstm_47/lstm_cell_185/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split
lstm_47/lstm_cell_185/SigmoidSigmoid$lstm_47/lstm_cell_185/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_47/lstm_cell_185/Sigmoid_1Sigmoid$lstm_47/lstm_cell_185/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_47/lstm_cell_185/mulMul#lstm_47/lstm_cell_185/Sigmoid_1:y:0lstm_47/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
lstm_47/lstm_cell_185/ReluRelu$lstm_47/lstm_cell_185/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¡
lstm_47/lstm_cell_185/mul_1Mul!lstm_47/lstm_cell_185/Sigmoid:y:0(lstm_47/lstm_cell_185/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_47/lstm_cell_185/add_1AddV2lstm_47/lstm_cell_185/mul:z:0lstm_47/lstm_cell_185/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_47/lstm_cell_185/Sigmoid_2Sigmoid$lstm_47/lstm_cell_185/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
lstm_47/lstm_cell_185/Relu_1Relulstm_47/lstm_cell_185/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¥
lstm_47/lstm_cell_185/mul_2Mul#lstm_47/lstm_cell_185/Sigmoid_2:y:0*lstm_47/lstm_cell_185/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ v
%lstm_47/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    f
$lstm_47/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_47/TensorArrayV2_1TensorListReserve.lstm_47/TensorArrayV2_1/element_shape:output:0-lstm_47/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒN
lstm_47/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_47/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ\
lstm_47/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ÷
lstm_47/whileWhile#lstm_47/while/loop_counter:output:0)lstm_47/while/maximum_iterations:output:0lstm_47/time:output:0 lstm_47/TensorArrayV2_1:handle:0lstm_47/zeros:output:0lstm_47/zeros_1:output:0 lstm_47/strided_slice_1:output:0?lstm_47/TensorArrayUnstack/TensorListFromTensor:output_handle:04lstm_47_lstm_cell_185_matmul_readvariableop_resource6lstm_47_lstm_cell_185_matmul_1_readvariableop_resource5lstm_47_lstm_cell_185_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_47_while_body_3534260*&
condR
lstm_47_while_cond_3534259*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
8lstm_47/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    î
*lstm_47/TensorArrayV2Stack/TensorListStackTensorListStacklstm_47/while:output:3Alstm_47/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0*
num_elementsp
lstm_47/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿi
lstm_47/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_47/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¯
lstm_47/strided_slice_3StridedSlice3lstm_47/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_47/strided_slice_3/stack:output:0(lstm_47/strided_slice_3/stack_1:output:0(lstm_47/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maskm
lstm_47/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ®
lstm_47/transpose_1	Transpose3lstm_47/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_47/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
lstm_47/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_21/MatMulMatMul lstm_47/strided_slice_3:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
reshape_8/ShapeShapedense_21/BiasAdd:output:0*
T0*
_output_shapes
:g
reshape_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_8/strided_sliceStridedSlicereshape_8/Shape:output:0&reshape_8/strided_slice/stack:output:0(reshape_8/strided_slice/stack_1:output:0(reshape_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_8/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :·
reshape_8/Reshape/shapePack reshape_8/strided_slice:output:0"reshape_8/Reshape/shape/1:output:0"reshape_8/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:
reshape_8/ReshapeReshapedense_21/BiasAdd:output:0 reshape_8/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
IdentityIdentityreshape_8/Reshape:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
NoOpNoOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp-^lstm_45/lstm_cell_183/BiasAdd/ReadVariableOp,^lstm_45/lstm_cell_183/MatMul/ReadVariableOp.^lstm_45/lstm_cell_183/MatMul_1/ReadVariableOp^lstm_45/while-^lstm_46/lstm_cell_184/BiasAdd/ReadVariableOp,^lstm_46/lstm_cell_184/MatMul/ReadVariableOp.^lstm_46/lstm_cell_184/MatMul_1/ReadVariableOp^lstm_46/while-^lstm_47/lstm_cell_185/BiasAdd/ReadVariableOp,^lstm_47/lstm_cell_185/MatMul/ReadVariableOp.^lstm_47/lstm_cell_185/MatMul_1/ReadVariableOp^lstm_47/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : 2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2\
,lstm_45/lstm_cell_183/BiasAdd/ReadVariableOp,lstm_45/lstm_cell_183/BiasAdd/ReadVariableOp2Z
+lstm_45/lstm_cell_183/MatMul/ReadVariableOp+lstm_45/lstm_cell_183/MatMul/ReadVariableOp2^
-lstm_45/lstm_cell_183/MatMul_1/ReadVariableOp-lstm_45/lstm_cell_183/MatMul_1/ReadVariableOp2
lstm_45/whilelstm_45/while2\
,lstm_46/lstm_cell_184/BiasAdd/ReadVariableOp,lstm_46/lstm_cell_184/BiasAdd/ReadVariableOp2Z
+lstm_46/lstm_cell_184/MatMul/ReadVariableOp+lstm_46/lstm_cell_184/MatMul/ReadVariableOp2^
-lstm_46/lstm_cell_184/MatMul_1/ReadVariableOp-lstm_46/lstm_cell_184/MatMul_1/ReadVariableOp2
lstm_46/whilelstm_46/while2\
,lstm_47/lstm_cell_185/BiasAdd/ReadVariableOp,lstm_47/lstm_cell_185/BiasAdd/ReadVariableOp2Z
+lstm_47/lstm_cell_185/MatMul/ReadVariableOp+lstm_47/lstm_cell_185/MatMul/ReadVariableOp2^
-lstm_47/lstm_cell_185/MatMul_1/ReadVariableOp-lstm_47/lstm_cell_185/MatMul_1/ReadVariableOp2
lstm_47/whilelstm_47/while:N J
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


_user_specified_namex
#
ñ
while_body_3531410
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_183_3531434_0:	0
while_lstm_cell_183_3531436_0:	 ,
while_lstm_cell_183_3531438_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_183_3531434:	.
while_lstm_cell_183_3531436:	 *
while_lstm_cell_183_3531438:	¢+while/lstm_cell_183/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¼
+while/lstm_cell_183/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_183_3531434_0while_lstm_cell_183_3531436_0while_lstm_cell_183_3531438_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_183_layer_call_and_return_conditional_losses_3531351Ý
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder4while/lstm_cell_183/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity4while/lstm_cell_183/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/Identity_5Identity4while/lstm_cell_183/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z

while/NoOpNoOp,^while/lstm_cell_183/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"<
while_lstm_cell_183_3531434while_lstm_cell_183_3531434_0"<
while_lstm_cell_183_3531436while_lstm_cell_183_3531436_0"<
while_lstm_cell_183_3531438while_lstm_cell_183_3531438_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2Z
+while/lstm_cell_183/StatefulPartitionedCall+while/lstm_cell_183/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
B
Ú

lstm_45_while_body_3533981,
(lstm_45_while_lstm_45_while_loop_counter2
.lstm_45_while_lstm_45_while_maximum_iterations
lstm_45_while_placeholder
lstm_45_while_placeholder_1
lstm_45_while_placeholder_2
lstm_45_while_placeholder_3+
'lstm_45_while_lstm_45_strided_slice_1_0g
clstm_45_while_tensorarrayv2read_tensorlistgetitem_lstm_45_tensorarrayunstack_tensorlistfromtensor_0O
<lstm_45_while_lstm_cell_183_matmul_readvariableop_resource_0:	Q
>lstm_45_while_lstm_cell_183_matmul_1_readvariableop_resource_0:	 L
=lstm_45_while_lstm_cell_183_biasadd_readvariableop_resource_0:	
lstm_45_while_identity
lstm_45_while_identity_1
lstm_45_while_identity_2
lstm_45_while_identity_3
lstm_45_while_identity_4
lstm_45_while_identity_5)
%lstm_45_while_lstm_45_strided_slice_1e
alstm_45_while_tensorarrayv2read_tensorlistgetitem_lstm_45_tensorarrayunstack_tensorlistfromtensorM
:lstm_45_while_lstm_cell_183_matmul_readvariableop_resource:	O
<lstm_45_while_lstm_cell_183_matmul_1_readvariableop_resource:	 J
;lstm_45_while_lstm_cell_183_biasadd_readvariableop_resource:	¢2lstm_45/while/lstm_cell_183/BiasAdd/ReadVariableOp¢1lstm_45/while/lstm_cell_183/MatMul/ReadVariableOp¢3lstm_45/while/lstm_cell_183/MatMul_1/ReadVariableOp
?lstm_45/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Î
1lstm_45/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_45_while_tensorarrayv2read_tensorlistgetitem_lstm_45_tensorarrayunstack_tensorlistfromtensor_0lstm_45_while_placeholderHlstm_45/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¯
1lstm_45/while/lstm_cell_183/MatMul/ReadVariableOpReadVariableOp<lstm_45_while_lstm_cell_183_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0Ô
"lstm_45/while/lstm_cell_183/MatMulMatMul8lstm_45/while/TensorArrayV2Read/TensorListGetItem:item:09lstm_45/while/lstm_cell_183/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
3lstm_45/while/lstm_cell_183/MatMul_1/ReadVariableOpReadVariableOp>lstm_45_while_lstm_cell_183_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0»
$lstm_45/while/lstm_cell_183/MatMul_1MatMullstm_45_while_placeholder_2;lstm_45/while/lstm_cell_183/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
lstm_45/while/lstm_cell_183/addAddV2,lstm_45/while/lstm_cell_183/MatMul:product:0.lstm_45/while/lstm_cell_183/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
2lstm_45/while/lstm_cell_183/BiasAdd/ReadVariableOpReadVariableOp=lstm_45_while_lstm_cell_183_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0Â
#lstm_45/while/lstm_cell_183/BiasAddBiasAdd#lstm_45/while/lstm_cell_183/add:z:0:lstm_45/while/lstm_cell_183/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
+lstm_45/while/lstm_cell_183/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
!lstm_45/while/lstm_cell_183/splitSplit4lstm_45/while/lstm_cell_183/split/split_dim:output:0,lstm_45/while/lstm_cell_183/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split
#lstm_45/while/lstm_cell_183/SigmoidSigmoid*lstm_45/while/lstm_cell_183/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
%lstm_45/while/lstm_cell_183/Sigmoid_1Sigmoid*lstm_45/while/lstm_cell_183/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
lstm_45/while/lstm_cell_183/mulMul)lstm_45/while/lstm_cell_183/Sigmoid_1:y:0lstm_45_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 lstm_45/while/lstm_cell_183/ReluRelu*lstm_45/while/lstm_cell_183/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ³
!lstm_45/while/lstm_cell_183/mul_1Mul'lstm_45/while/lstm_cell_183/Sigmoid:y:0.lstm_45/while/lstm_cell_183/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¨
!lstm_45/while/lstm_cell_183/add_1AddV2#lstm_45/while/lstm_cell_183/mul:z:0%lstm_45/while/lstm_cell_183/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
%lstm_45/while/lstm_cell_183/Sigmoid_2Sigmoid*lstm_45/while/lstm_cell_183/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"lstm_45/while/lstm_cell_183/Relu_1Relu%lstm_45/while/lstm_cell_183/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ·
!lstm_45/while/lstm_cell_183/mul_2Mul)lstm_45/while/lstm_cell_183/Sigmoid_2:y:00lstm_45/while/lstm_cell_183/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ æ
2lstm_45/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_45_while_placeholder_1lstm_45_while_placeholder%lstm_45/while/lstm_cell_183/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒU
lstm_45/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_45/while/addAddV2lstm_45_while_placeholderlstm_45/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_45/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_45/while/add_1AddV2(lstm_45_while_lstm_45_while_loop_counterlstm_45/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_45/while/IdentityIdentitylstm_45/while/add_1:z:0^lstm_45/while/NoOp*
T0*
_output_shapes
: 
lstm_45/while/Identity_1Identity.lstm_45_while_lstm_45_while_maximum_iterations^lstm_45/while/NoOp*
T0*
_output_shapes
: q
lstm_45/while/Identity_2Identitylstm_45/while/add:z:0^lstm_45/while/NoOp*
T0*
_output_shapes
: 
lstm_45/while/Identity_3IdentityBlstm_45/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_45/while/NoOp*
T0*
_output_shapes
: 
lstm_45/while/Identity_4Identity%lstm_45/while/lstm_cell_183/mul_2:z:0^lstm_45/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_45/while/Identity_5Identity%lstm_45/while/lstm_cell_183/add_1:z:0^lstm_45/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ó
lstm_45/while/NoOpNoOp3^lstm_45/while/lstm_cell_183/BiasAdd/ReadVariableOp2^lstm_45/while/lstm_cell_183/MatMul/ReadVariableOp4^lstm_45/while/lstm_cell_183/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_45_while_identitylstm_45/while/Identity:output:0"=
lstm_45_while_identity_1!lstm_45/while/Identity_1:output:0"=
lstm_45_while_identity_2!lstm_45/while/Identity_2:output:0"=
lstm_45_while_identity_3!lstm_45/while/Identity_3:output:0"=
lstm_45_while_identity_4!lstm_45/while/Identity_4:output:0"=
lstm_45_while_identity_5!lstm_45/while/Identity_5:output:0"P
%lstm_45_while_lstm_45_strided_slice_1'lstm_45_while_lstm_45_strided_slice_1_0"|
;lstm_45_while_lstm_cell_183_biasadd_readvariableop_resource=lstm_45_while_lstm_cell_183_biasadd_readvariableop_resource_0"~
<lstm_45_while_lstm_cell_183_matmul_1_readvariableop_resource>lstm_45_while_lstm_cell_183_matmul_1_readvariableop_resource_0"z
:lstm_45_while_lstm_cell_183_matmul_readvariableop_resource<lstm_45_while_lstm_cell_183_matmul_readvariableop_resource_0"È
alstm_45_while_tensorarrayv2read_tensorlistgetitem_lstm_45_tensorarrayunstack_tensorlistfromtensorclstm_45_while_tensorarrayv2read_tensorlistgetitem_lstm_45_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2h
2lstm_45/while/lstm_cell_183/BiasAdd/ReadVariableOp2lstm_45/while/lstm_cell_183/BiasAdd/ReadVariableOp2f
1lstm_45/while/lstm_cell_183/MatMul/ReadVariableOp1lstm_45/while/lstm_cell_183/MatMul/ReadVariableOp2j
3lstm_45/while/lstm_cell_183/MatMul_1/ReadVariableOp3lstm_45/while/lstm_cell_183/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
ß

J__inference_lstm_cell_184_layer_call_and_return_conditional_losses_3536449

inputs
states_0
states_11
matmul_readvariableop_resource:	 3
 matmul_1_readvariableop_resource:	 .
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 *
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ N
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/1
º
È
while_cond_3531218
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3531218___redundant_placeholder05
1while_while_cond_3531218___redundant_placeholder15
1while_while_cond_3531218___redundant_placeholder25
1while_while_cond_3531218___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
ß

J__inference_lstm_cell_184_layer_call_and_return_conditional_losses_3536417

inputs
states_0
states_11
matmul_readvariableop_resource:	 3
 matmul_1_readvariableop_resource:	 .
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 *
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ N
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/1
³Q
û
'rnn_model_21_lstm_45_while_body_3530759F
Brnn_model_21_lstm_45_while_rnn_model_21_lstm_45_while_loop_counterL
Hrnn_model_21_lstm_45_while_rnn_model_21_lstm_45_while_maximum_iterations*
&rnn_model_21_lstm_45_while_placeholder,
(rnn_model_21_lstm_45_while_placeholder_1,
(rnn_model_21_lstm_45_while_placeholder_2,
(rnn_model_21_lstm_45_while_placeholder_3E
Arnn_model_21_lstm_45_while_rnn_model_21_lstm_45_strided_slice_1_0
}rnn_model_21_lstm_45_while_tensorarrayv2read_tensorlistgetitem_rnn_model_21_lstm_45_tensorarrayunstack_tensorlistfromtensor_0\
Irnn_model_21_lstm_45_while_lstm_cell_183_matmul_readvariableop_resource_0:	^
Krnn_model_21_lstm_45_while_lstm_cell_183_matmul_1_readvariableop_resource_0:	 Y
Jrnn_model_21_lstm_45_while_lstm_cell_183_biasadd_readvariableop_resource_0:	'
#rnn_model_21_lstm_45_while_identity)
%rnn_model_21_lstm_45_while_identity_1)
%rnn_model_21_lstm_45_while_identity_2)
%rnn_model_21_lstm_45_while_identity_3)
%rnn_model_21_lstm_45_while_identity_4)
%rnn_model_21_lstm_45_while_identity_5C
?rnn_model_21_lstm_45_while_rnn_model_21_lstm_45_strided_slice_1
{rnn_model_21_lstm_45_while_tensorarrayv2read_tensorlistgetitem_rnn_model_21_lstm_45_tensorarrayunstack_tensorlistfromtensorZ
Grnn_model_21_lstm_45_while_lstm_cell_183_matmul_readvariableop_resource:	\
Irnn_model_21_lstm_45_while_lstm_cell_183_matmul_1_readvariableop_resource:	 W
Hrnn_model_21_lstm_45_while_lstm_cell_183_biasadd_readvariableop_resource:	¢?rnn_model_21/lstm_45/while/lstm_cell_183/BiasAdd/ReadVariableOp¢>rnn_model_21/lstm_45/while/lstm_cell_183/MatMul/ReadVariableOp¢@rnn_model_21/lstm_45/while/lstm_cell_183/MatMul_1/ReadVariableOp
Lrnn_model_21/lstm_45/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
>rnn_model_21/lstm_45/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}rnn_model_21_lstm_45_while_tensorarrayv2read_tensorlistgetitem_rnn_model_21_lstm_45_tensorarrayunstack_tensorlistfromtensor_0&rnn_model_21_lstm_45_while_placeholderUrnn_model_21/lstm_45/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0É
>rnn_model_21/lstm_45/while/lstm_cell_183/MatMul/ReadVariableOpReadVariableOpIrnn_model_21_lstm_45_while_lstm_cell_183_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0û
/rnn_model_21/lstm_45/while/lstm_cell_183/MatMulMatMulErnn_model_21/lstm_45/while/TensorArrayV2Read/TensorListGetItem:item:0Frnn_model_21/lstm_45/while/lstm_cell_183/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ
@rnn_model_21/lstm_45/while/lstm_cell_183/MatMul_1/ReadVariableOpReadVariableOpKrnn_model_21_lstm_45_while_lstm_cell_183_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0â
1rnn_model_21/lstm_45/while/lstm_cell_183/MatMul_1MatMul(rnn_model_21_lstm_45_while_placeholder_2Hrnn_model_21/lstm_45/while/lstm_cell_183/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
,rnn_model_21/lstm_45/while/lstm_cell_183/addAddV29rnn_model_21/lstm_45/while/lstm_cell_183/MatMul:product:0;rnn_model_21/lstm_45/while/lstm_cell_183/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ
?rnn_model_21/lstm_45/while/lstm_cell_183/BiasAdd/ReadVariableOpReadVariableOpJrnn_model_21_lstm_45_while_lstm_cell_183_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0é
0rnn_model_21/lstm_45/while/lstm_cell_183/BiasAddBiasAdd0rnn_model_21/lstm_45/while/lstm_cell_183/add:z:0Grnn_model_21/lstm_45/while/lstm_cell_183/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
8rnn_model_21/lstm_45/while/lstm_cell_183/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :±
.rnn_model_21/lstm_45/while/lstm_cell_183/splitSplitArnn_model_21/lstm_45/while/lstm_cell_183/split/split_dim:output:09rnn_model_21/lstm_45/while/lstm_cell_183/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split¦
0rnn_model_21/lstm_45/while/lstm_cell_183/SigmoidSigmoid7rnn_model_21/lstm_45/while/lstm_cell_183/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¨
2rnn_model_21/lstm_45/while/lstm_cell_183/Sigmoid_1Sigmoid7rnn_model_21/lstm_45/while/lstm_cell_183/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ç
,rnn_model_21/lstm_45/while/lstm_cell_183/mulMul6rnn_model_21/lstm_45/while/lstm_cell_183/Sigmoid_1:y:0(rnn_model_21_lstm_45_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
-rnn_model_21/lstm_45/while/lstm_cell_183/ReluRelu7rnn_model_21/lstm_45/while/lstm_cell_183/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ú
.rnn_model_21/lstm_45/while/lstm_cell_183/mul_1Mul4rnn_model_21/lstm_45/while/lstm_cell_183/Sigmoid:y:0;rnn_model_21/lstm_45/while/lstm_cell_183/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ï
.rnn_model_21/lstm_45/while/lstm_cell_183/add_1AddV20rnn_model_21/lstm_45/while/lstm_cell_183/mul:z:02rnn_model_21/lstm_45/while/lstm_cell_183/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¨
2rnn_model_21/lstm_45/while/lstm_cell_183/Sigmoid_2Sigmoid7rnn_model_21/lstm_45/while/lstm_cell_183/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
/rnn_model_21/lstm_45/while/lstm_cell_183/Relu_1Relu2rnn_model_21/lstm_45/while/lstm_cell_183/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Þ
.rnn_model_21/lstm_45/while/lstm_cell_183/mul_2Mul6rnn_model_21/lstm_45/while/lstm_cell_183/Sigmoid_2:y:0=rnn_model_21/lstm_45/while/lstm_cell_183/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
?rnn_model_21/lstm_45/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(rnn_model_21_lstm_45_while_placeholder_1&rnn_model_21_lstm_45_while_placeholder2rnn_model_21/lstm_45/while/lstm_cell_183/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒb
 rnn_model_21/lstm_45/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
rnn_model_21/lstm_45/while/addAddV2&rnn_model_21_lstm_45_while_placeholder)rnn_model_21/lstm_45/while/add/y:output:0*
T0*
_output_shapes
: d
"rnn_model_21/lstm_45/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :»
 rnn_model_21/lstm_45/while/add_1AddV2Brnn_model_21_lstm_45_while_rnn_model_21_lstm_45_while_loop_counter+rnn_model_21/lstm_45/while/add_1/y:output:0*
T0*
_output_shapes
: 
#rnn_model_21/lstm_45/while/IdentityIdentity$rnn_model_21/lstm_45/while/add_1:z:0 ^rnn_model_21/lstm_45/while/NoOp*
T0*
_output_shapes
: ¾
%rnn_model_21/lstm_45/while/Identity_1IdentityHrnn_model_21_lstm_45_while_rnn_model_21_lstm_45_while_maximum_iterations ^rnn_model_21/lstm_45/while/NoOp*
T0*
_output_shapes
: 
%rnn_model_21/lstm_45/while/Identity_2Identity"rnn_model_21/lstm_45/while/add:z:0 ^rnn_model_21/lstm_45/while/NoOp*
T0*
_output_shapes
: Å
%rnn_model_21/lstm_45/while/Identity_3IdentityOrnn_model_21/lstm_45/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^rnn_model_21/lstm_45/while/NoOp*
T0*
_output_shapes
: ¹
%rnn_model_21/lstm_45/while/Identity_4Identity2rnn_model_21/lstm_45/while/lstm_cell_183/mul_2:z:0 ^rnn_model_21/lstm_45/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¹
%rnn_model_21/lstm_45/while/Identity_5Identity2rnn_model_21/lstm_45/while/lstm_cell_183/add_1:z:0 ^rnn_model_21/lstm_45/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ §
rnn_model_21/lstm_45/while/NoOpNoOp@^rnn_model_21/lstm_45/while/lstm_cell_183/BiasAdd/ReadVariableOp?^rnn_model_21/lstm_45/while/lstm_cell_183/MatMul/ReadVariableOpA^rnn_model_21/lstm_45/while/lstm_cell_183/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "S
#rnn_model_21_lstm_45_while_identity,rnn_model_21/lstm_45/while/Identity:output:0"W
%rnn_model_21_lstm_45_while_identity_1.rnn_model_21/lstm_45/while/Identity_1:output:0"W
%rnn_model_21_lstm_45_while_identity_2.rnn_model_21/lstm_45/while/Identity_2:output:0"W
%rnn_model_21_lstm_45_while_identity_3.rnn_model_21/lstm_45/while/Identity_3:output:0"W
%rnn_model_21_lstm_45_while_identity_4.rnn_model_21/lstm_45/while/Identity_4:output:0"W
%rnn_model_21_lstm_45_while_identity_5.rnn_model_21/lstm_45/while/Identity_5:output:0"
Hrnn_model_21_lstm_45_while_lstm_cell_183_biasadd_readvariableop_resourceJrnn_model_21_lstm_45_while_lstm_cell_183_biasadd_readvariableop_resource_0"
Irnn_model_21_lstm_45_while_lstm_cell_183_matmul_1_readvariableop_resourceKrnn_model_21_lstm_45_while_lstm_cell_183_matmul_1_readvariableop_resource_0"
Grnn_model_21_lstm_45_while_lstm_cell_183_matmul_readvariableop_resourceIrnn_model_21_lstm_45_while_lstm_cell_183_matmul_readvariableop_resource_0"
?rnn_model_21_lstm_45_while_rnn_model_21_lstm_45_strided_slice_1Arnn_model_21_lstm_45_while_rnn_model_21_lstm_45_strided_slice_1_0"ü
{rnn_model_21_lstm_45_while_tensorarrayv2read_tensorlistgetitem_rnn_model_21_lstm_45_tensorarrayunstack_tensorlistfromtensor}rnn_model_21_lstm_45_while_tensorarrayv2read_tensorlistgetitem_rnn_model_21_lstm_45_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2
?rnn_model_21/lstm_45/while/lstm_cell_183/BiasAdd/ReadVariableOp?rnn_model_21/lstm_45/while/lstm_cell_183/BiasAdd/ReadVariableOp2
>rnn_model_21/lstm_45/while/lstm_cell_183/MatMul/ReadVariableOp>rnn_model_21/lstm_45/while/lstm_cell_183/MatMul/ReadVariableOp2
@rnn_model_21/lstm_45/while/lstm_cell_183/MatMul_1/ReadVariableOp@rnn_model_21/lstm_45/while/lstm_cell_183/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
è


.__inference_rnn_model_21_layer_call_fn_3533333
input_1
unknown:	
	unknown_0:	 
	unknown_1:	
	unknown_2:	 
	unknown_3:	 
	unknown_4:	
	unknown_5:	 
	unknown_6:	 
	unknown_7:	
	unknown_8: 
	unknown_9:
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_rnn_model_21_layer_call_and_return_conditional_losses_3533281s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1
ß

J__inference_lstm_cell_185_layer_call_and_return_conditional_losses_3536515

inputs
states_0
states_11
matmul_readvariableop_resource:	 3
 matmul_1_readvariableop_resource:	 .
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 *
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ N
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/1
çK
¢
D__inference_lstm_47_layer_call_and_return_conditional_losses_3532643

inputs?
,lstm_cell_185_matmul_readvariableop_resource:	 A
.lstm_cell_185_matmul_1_readvariableop_resource:	 <
-lstm_cell_185_biasadd_readvariableop_resource:	
identity¢$lstm_cell_185/BiasAdd/ReadVariableOp¢#lstm_cell_185/MatMul/ReadVariableOp¢%lstm_cell_185/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿ D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask
#lstm_cell_185/MatMul/ReadVariableOpReadVariableOp,lstm_cell_185_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_185/MatMulMatMulstrided_slice_2:output:0+lstm_cell_185/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%lstm_cell_185/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_185_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_185/MatMul_1MatMulzeros:output:0-lstm_cell_185/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_185/addAddV2lstm_cell_185/MatMul:product:0 lstm_cell_185/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_185/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_185_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_185/BiasAddBiasAddlstm_cell_185/add:z:0,lstm_cell_185/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
lstm_cell_185/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :à
lstm_cell_185/splitSplit&lstm_cell_185/split/split_dim:output:0lstm_cell_185/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitp
lstm_cell_185/SigmoidSigmoidlstm_cell_185/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
lstm_cell_185/Sigmoid_1Sigmoidlstm_cell_185/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y
lstm_cell_185/mulMullstm_cell_185/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
lstm_cell_185/ReluRelulstm_cell_185/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_185/mul_1Mullstm_cell_185/Sigmoid:y:0 lstm_cell_185/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
lstm_cell_185/add_1AddV2lstm_cell_185/mul:z:0lstm_cell_185/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
lstm_cell_185/Sigmoid_2Sigmoidlstm_cell_185/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ g
lstm_cell_185/Relu_1Relulstm_cell_185/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_185/mul_2Mullstm_cell_185/Sigmoid_2:y:0"lstm_cell_185/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_185_matmul_readvariableop_resource.lstm_cell_185_matmul_1_readvariableop_resource-lstm_cell_185_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3532558*
condR
while_cond_3532557*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ã
NoOpNoOp%^lstm_cell_185/BiasAdd/ReadVariableOp$^lstm_cell_185/MatMul/ReadVariableOp&^lstm_cell_185/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
 : : : 2L
$lstm_cell_185/BiasAdd/ReadVariableOp$lstm_cell_185/BiasAdd/ReadVariableOp2J
#lstm_cell_185/MatMul/ReadVariableOp#lstm_cell_185/MatMul/ReadVariableOp2N
%lstm_cell_185/MatMul_1/ReadVariableOp%lstm_cell_185/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
 
_user_specified_nameinputs
Ñ8
Ú
while_body_3534606
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_183_matmul_readvariableop_resource_0:	I
6while_lstm_cell_183_matmul_1_readvariableop_resource_0:	 D
5while_lstm_cell_183_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_183_matmul_readvariableop_resource:	G
4while_lstm_cell_183_matmul_1_readvariableop_resource:	 B
3while_lstm_cell_183_biasadd_readvariableop_resource:	¢*while/lstm_cell_183/BiasAdd/ReadVariableOp¢)while/lstm_cell_183/MatMul/ReadVariableOp¢+while/lstm_cell_183/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
)while/lstm_cell_183/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_183_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0¼
while/lstm_cell_183/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_183/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
+while/lstm_cell_183/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_183_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0£
while/lstm_cell_183/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_183/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
while/lstm_cell_183/addAddV2$while/lstm_cell_183/MatMul:product:0&while/lstm_cell_183/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*while/lstm_cell_183/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_183_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0ª
while/lstm_cell_183/BiasAddBiasAddwhile/lstm_cell_183/add:z:02while/lstm_cell_183/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
#while/lstm_cell_183/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ò
while/lstm_cell_183/splitSplit,while/lstm_cell_183/split/split_dim:output:0$while/lstm_cell_183/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split|
while/lstm_cell_183/SigmoidSigmoid"while/lstm_cell_183/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
while/lstm_cell_183/Sigmoid_1Sigmoid"while/lstm_cell_183/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_183/mulMul!while/lstm_cell_183/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ v
while/lstm_cell_183/ReluRelu"while/lstm_cell_183/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_183/mul_1Mulwhile/lstm_cell_183/Sigmoid:y:0&while/lstm_cell_183/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_183/add_1AddV2while/lstm_cell_183/mul:z:0while/lstm_cell_183/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
while/lstm_cell_183/Sigmoid_2Sigmoid"while/lstm_cell_183/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ s
while/lstm_cell_183/Relu_1Reluwhile/lstm_cell_183/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_183/mul_2Mul!while/lstm_cell_183/Sigmoid_2:y:0(while/lstm_cell_183/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Æ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_183/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_183/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
while/Identity_5Identitywhile/lstm_cell_183/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ó

while/NoOpNoOp+^while/lstm_cell_183/BiasAdd/ReadVariableOp*^while/lstm_cell_183/MatMul/ReadVariableOp,^while/lstm_cell_183/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_183_biasadd_readvariableop_resource5while_lstm_cell_183_biasadd_readvariableop_resource_0"n
4while_lstm_cell_183_matmul_1_readvariableop_resource6while_lstm_cell_183_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_183_matmul_readvariableop_resource4while_lstm_cell_183_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2X
*while/lstm_cell_183/BiasAdd/ReadVariableOp*while/lstm_cell_183/BiasAdd/ReadVariableOp2V
)while/lstm_cell_183/MatMul/ReadVariableOp)while/lstm_cell_183/MatMul/ReadVariableOp2Z
+while/lstm_cell_183/MatMul_1/ReadVariableOp+while/lstm_cell_183/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
º
È
while_cond_3531409
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3531409___redundant_placeholder05
1while_while_cond_3531409___redundant_placeholder15
1while_while_cond_3531409___redundant_placeholder25
1while_while_cond_3531409___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
í9
Ú
while_body_3532558
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_185_matmul_readvariableop_resource_0:	 I
6while_lstm_cell_185_matmul_1_readvariableop_resource_0:	 D
5while_lstm_cell_185_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_185_matmul_readvariableop_resource:	 G
4while_lstm_cell_185_matmul_1_readvariableop_resource:	 B
3while_lstm_cell_185_biasadd_readvariableop_resource:	¢*while/lstm_cell_185/BiasAdd/ReadVariableOp¢)while/lstm_cell_185/MatMul/ReadVariableOp¢+while/lstm_cell_185/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0
)while/lstm_cell_185/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_185_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype0¼
while/lstm_cell_185/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_185/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
+while/lstm_cell_185/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_185_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0£
while/lstm_cell_185/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_185/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
while/lstm_cell_185/addAddV2$while/lstm_cell_185/MatMul:product:0&while/lstm_cell_185/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*while/lstm_cell_185/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_185_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0ª
while/lstm_cell_185/BiasAddBiasAddwhile/lstm_cell_185/add:z:02while/lstm_cell_185/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
#while/lstm_cell_185/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ò
while/lstm_cell_185/splitSplit,while/lstm_cell_185/split/split_dim:output:0$while/lstm_cell_185/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split|
while/lstm_cell_185/SigmoidSigmoid"while/lstm_cell_185/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
while/lstm_cell_185/Sigmoid_1Sigmoid"while/lstm_cell_185/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_185/mulMul!while/lstm_cell_185/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ v
while/lstm_cell_185/ReluRelu"while/lstm_cell_185/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_185/mul_1Mulwhile/lstm_cell_185/Sigmoid:y:0&while/lstm_cell_185/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_185/add_1AddV2while/lstm_cell_185/mul:z:0while/lstm_cell_185/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
while/lstm_cell_185/Sigmoid_2Sigmoid"while/lstm_cell_185/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ s
while/lstm_cell_185/Relu_1Reluwhile/lstm_cell_185/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_185/mul_2Mul!while/lstm_cell_185/Sigmoid_2:y:0(while/lstm_cell_185/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : î
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_185/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_185/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
while/Identity_5Identitywhile/lstm_cell_185/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ó

while/NoOpNoOp+^while/lstm_cell_185/BiasAdd/ReadVariableOp*^while/lstm_cell_185/MatMul/ReadVariableOp,^while/lstm_cell_185/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_185_biasadd_readvariableop_resource5while_lstm_cell_185_biasadd_readvariableop_resource_0"n
4while_lstm_cell_185_matmul_1_readvariableop_resource6while_lstm_cell_185_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_185_matmul_readvariableop_resource4while_lstm_cell_185_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2X
*while/lstm_cell_185/BiasAdd/ReadVariableOp*while/lstm_cell_185/BiasAdd/ReadVariableOp2V
)while/lstm_cell_185/MatMul/ReadVariableOp)while/lstm_cell_185/MatMul/ReadVariableOp2Z
+while/lstm_cell_185/MatMul_1/ReadVariableOp+while/lstm_cell_185/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
K
¤
D__inference_lstm_46_layer_call_and_return_conditional_losses_3535306
inputs_0?
,lstm_cell_184_matmul_readvariableop_resource:	 A
.lstm_cell_184_matmul_1_readvariableop_resource:	 <
-lstm_cell_184_biasadd_readvariableop_resource:	
identity¢$lstm_cell_184/BiasAdd/ReadVariableOp¢#lstm_cell_184/MatMul/ReadVariableOp¢%lstm_cell_184/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask
#lstm_cell_184/MatMul/ReadVariableOpReadVariableOp,lstm_cell_184_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_184/MatMulMatMulstrided_slice_2:output:0+lstm_cell_184/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%lstm_cell_184/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_184_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_184/MatMul_1MatMulzeros:output:0-lstm_cell_184/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_184/addAddV2lstm_cell_184/MatMul:product:0 lstm_cell_184/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_184/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_184_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_184/BiasAddBiasAddlstm_cell_184/add:z:0,lstm_cell_184/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
lstm_cell_184/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :à
lstm_cell_184/splitSplit&lstm_cell_184/split/split_dim:output:0lstm_cell_184/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitp
lstm_cell_184/SigmoidSigmoidlstm_cell_184/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
lstm_cell_184/Sigmoid_1Sigmoidlstm_cell_184/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y
lstm_cell_184/mulMullstm_cell_184/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
lstm_cell_184/ReluRelulstm_cell_184/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_184/mul_1Mullstm_cell_184/Sigmoid:y:0 lstm_cell_184/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
lstm_cell_184/add_1AddV2lstm_cell_184/mul:z:0lstm_cell_184/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
lstm_cell_184/Sigmoid_2Sigmoidlstm_cell_184/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ g
lstm_cell_184/Relu_1Relulstm_cell_184/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_184/mul_2Mullstm_cell_184/Sigmoid_2:y:0"lstm_cell_184/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_184_matmul_readvariableop_resource.lstm_cell_184_matmul_1_readvariableop_resource-lstm_cell_184_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3535222*
condR
while_cond_3535221*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ã
NoOpNoOp%^lstm_cell_184/BiasAdd/ReadVariableOp$^lstm_cell_184/MatMul/ReadVariableOp&^lstm_cell_184/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : 2L
$lstm_cell_184/BiasAdd/ReadVariableOp$lstm_cell_184/BiasAdd/ReadVariableOp2J
#lstm_cell_184/MatMul/ReadVariableOp#lstm_cell_184/MatMul/ReadVariableOp2N
%lstm_cell_184/MatMul_1/ReadVariableOp%lstm_cell_184/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0
ù
¶
)__inference_lstm_47_layer_call_fn_3535636

inputs
unknown:	 
	unknown_0:	 
	unknown_1:	
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_47_layer_call_and_return_conditional_losses_3532882o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
 : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
 
_user_specified_nameinputs

î
I__inference_rnn_model_21_layer_call_and_return_conditional_losses_3533364
input_1"
lstm_45_3533336:	"
lstm_45_3533338:	 
lstm_45_3533340:	"
lstm_46_3533343:	 "
lstm_46_3533345:	 
lstm_46_3533347:	"
lstm_47_3533350:	 "
lstm_47_3533352:	 
lstm_47_3533354:	"
dense_21_3533357: 
dense_21_3533359:
identity¢ dense_21/StatefulPartitionedCall¢lstm_45/StatefulPartitionedCall¢lstm_46/StatefulPartitionedCall¢lstm_47/StatefulPartitionedCall
lstm_45/StatefulPartitionedCallStatefulPartitionedCallinput_1lstm_45_3533336lstm_45_3533338lstm_45_3533340*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_45_layer_call_and_return_conditional_losses_3532341¨
lstm_46/StatefulPartitionedCallStatefulPartitionedCall(lstm_45/StatefulPartitionedCall:output:0lstm_46_3533343lstm_46_3533345lstm_46_3533347*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_46_layer_call_and_return_conditional_losses_3532491¤
lstm_47/StatefulPartitionedCallStatefulPartitionedCall(lstm_46/StatefulPartitionedCall:output:0lstm_47_3533350lstm_47_3533352lstm_47_3533354*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_47_layer_call_and_return_conditional_losses_3532643
 dense_21/StatefulPartitionedCallStatefulPartitionedCall(lstm_47/StatefulPartitionedCall:output:0dense_21_3533357dense_21_3533359*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_3532661â
reshape_8/PartitionedCallPartitionedCall)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reshape_8_layer_call_and_return_conditional_losses_3532680u
IdentityIdentity"reshape_8/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿÏ
NoOpNoOp!^dense_21/StatefulPartitionedCall ^lstm_45/StatefulPartitionedCall ^lstm_46/StatefulPartitionedCall ^lstm_47/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : 2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2B
lstm_45/StatefulPartitionedCalllstm_45/StatefulPartitionedCall2B
lstm_46/StatefulPartitionedCalllstm_46/StatefulPartitionedCall2B
lstm_47/StatefulPartitionedCalllstm_47/StatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1
Ñ8
Ú
while_body_3534463
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_183_matmul_readvariableop_resource_0:	I
6while_lstm_cell_183_matmul_1_readvariableop_resource_0:	 D
5while_lstm_cell_183_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_183_matmul_readvariableop_resource:	G
4while_lstm_cell_183_matmul_1_readvariableop_resource:	 B
3while_lstm_cell_183_biasadd_readvariableop_resource:	¢*while/lstm_cell_183/BiasAdd/ReadVariableOp¢)while/lstm_cell_183/MatMul/ReadVariableOp¢+while/lstm_cell_183/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
)while/lstm_cell_183/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_183_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0¼
while/lstm_cell_183/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_183/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
+while/lstm_cell_183/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_183_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0£
while/lstm_cell_183/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_183/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
while/lstm_cell_183/addAddV2$while/lstm_cell_183/MatMul:product:0&while/lstm_cell_183/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*while/lstm_cell_183/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_183_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0ª
while/lstm_cell_183/BiasAddBiasAddwhile/lstm_cell_183/add:z:02while/lstm_cell_183/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
#while/lstm_cell_183/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ò
while/lstm_cell_183/splitSplit,while/lstm_cell_183/split/split_dim:output:0$while/lstm_cell_183/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split|
while/lstm_cell_183/SigmoidSigmoid"while/lstm_cell_183/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
while/lstm_cell_183/Sigmoid_1Sigmoid"while/lstm_cell_183/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_183/mulMul!while/lstm_cell_183/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ v
while/lstm_cell_183/ReluRelu"while/lstm_cell_183/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_183/mul_1Mulwhile/lstm_cell_183/Sigmoid:y:0&while/lstm_cell_183/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_183/add_1AddV2while/lstm_cell_183/mul:z:0while/lstm_cell_183/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
while/lstm_cell_183/Sigmoid_2Sigmoid"while/lstm_cell_183/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ s
while/lstm_cell_183/Relu_1Reluwhile/lstm_cell_183/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_183/mul_2Mul!while/lstm_cell_183/Sigmoid_2:y:0(while/lstm_cell_183/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Æ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_183/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_183/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
while/Identity_5Identitywhile/lstm_cell_183/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ó

while/NoOpNoOp+^while/lstm_cell_183/BiasAdd/ReadVariableOp*^while/lstm_cell_183/MatMul/ReadVariableOp,^while/lstm_cell_183/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_183_biasadd_readvariableop_resource5while_lstm_cell_183_biasadd_readvariableop_resource_0"n
4while_lstm_cell_183_matmul_1_readvariableop_resource6while_lstm_cell_183_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_183_matmul_readvariableop_resource4while_lstm_cell_183_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2X
*while/lstm_cell_183/BiasAdd/ReadVariableOp*while/lstm_cell_183/BiasAdd/ReadVariableOp2V
)while/lstm_cell_183/MatMul/ReadVariableOp)while/lstm_cell_183/MatMul/ReadVariableOp2Z
+while/lstm_cell_183/MatMul_1/ReadVariableOp+while/lstm_cell_183/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 

î
I__inference_rnn_model_21_layer_call_and_return_conditional_losses_3533395
input_1"
lstm_45_3533367:	"
lstm_45_3533369:	 
lstm_45_3533371:	"
lstm_46_3533374:	 "
lstm_46_3533376:	 
lstm_46_3533378:	"
lstm_47_3533381:	 "
lstm_47_3533383:	 
lstm_47_3533385:	"
dense_21_3533388: 
dense_21_3533390:
identity¢ dense_21/StatefulPartitionedCall¢lstm_45/StatefulPartitionedCall¢lstm_46/StatefulPartitionedCall¢lstm_47/StatefulPartitionedCall
lstm_45/StatefulPartitionedCallStatefulPartitionedCallinput_1lstm_45_3533367lstm_45_3533369lstm_45_3533371*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_45_layer_call_and_return_conditional_losses_3533212¨
lstm_46/StatefulPartitionedCallStatefulPartitionedCall(lstm_45/StatefulPartitionedCall:output:0lstm_46_3533374lstm_46_3533376lstm_46_3533378*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_46_layer_call_and_return_conditional_losses_3533047¤
lstm_47/StatefulPartitionedCallStatefulPartitionedCall(lstm_46/StatefulPartitionedCall:output:0lstm_47_3533381lstm_47_3533383lstm_47_3533385*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_47_layer_call_and_return_conditional_losses_3532882
 dense_21/StatefulPartitionedCallStatefulPartitionedCall(lstm_47/StatefulPartitionedCall:output:0dense_21_3533388dense_21_3533390*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_3532661â
reshape_8/PartitionedCallPartitionedCall)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reshape_8_layer_call_and_return_conditional_losses_3532680u
IdentityIdentity"reshape_8/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿÏ
NoOpNoOp!^dense_21/StatefulPartitionedCall ^lstm_45/StatefulPartitionedCall ^lstm_46/StatefulPartitionedCall ^lstm_47/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : 2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2B
lstm_45/StatefulPartitionedCalllstm_45/StatefulPartitionedCall2B
lstm_46/StatefulPartitionedCalllstm_46/StatefulPartitionedCall2B
lstm_47/StatefulPartitionedCalllstm_47/StatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1
º
È
while_cond_3535840
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3535840___redundant_placeholder05
1while_while_cond_3535840___redundant_placeholder15
1while_while_cond_3535840___redundant_placeholder25
1while_while_cond_3535840___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
áJ
¢
D__inference_lstm_46_layer_call_and_return_conditional_losses_3535449

inputs?
,lstm_cell_184_matmul_readvariableop_resource:	 A
.lstm_cell_184_matmul_1_readvariableop_resource:	 <
-lstm_cell_184_biasadd_readvariableop_resource:	
identity¢$lstm_cell_184/BiasAdd/ReadVariableOp¢#lstm_cell_184/MatMul/ReadVariableOp¢%lstm_cell_184/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿ D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask
#lstm_cell_184/MatMul/ReadVariableOpReadVariableOp,lstm_cell_184_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_184/MatMulMatMulstrided_slice_2:output:0+lstm_cell_184/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%lstm_cell_184/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_184_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_184/MatMul_1MatMulzeros:output:0-lstm_cell_184/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_184/addAddV2lstm_cell_184/MatMul:product:0 lstm_cell_184/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_184/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_184_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_184/BiasAddBiasAddlstm_cell_184/add:z:0,lstm_cell_184/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
lstm_cell_184/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :à
lstm_cell_184/splitSplit&lstm_cell_184/split/split_dim:output:0lstm_cell_184/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitp
lstm_cell_184/SigmoidSigmoidlstm_cell_184/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
lstm_cell_184/Sigmoid_1Sigmoidlstm_cell_184/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y
lstm_cell_184/mulMullstm_cell_184/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
lstm_cell_184/ReluRelulstm_cell_184/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_184/mul_1Mullstm_cell_184/Sigmoid:y:0 lstm_cell_184/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
lstm_cell_184/add_1AddV2lstm_cell_184/mul:z:0lstm_cell_184/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
lstm_cell_184/Sigmoid_2Sigmoidlstm_cell_184/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ g
lstm_cell_184/Relu_1Relulstm_cell_184/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_184/mul_2Mullstm_cell_184/Sigmoid_2:y:0"lstm_cell_184/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_184_matmul_readvariableop_resource.lstm_cell_184_matmul_1_readvariableop_resource-lstm_cell_184_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3535365*
condR
while_cond_3535364*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 Ã
NoOpNoOp%^lstm_cell_184/BiasAdd/ReadVariableOp$^lstm_cell_184/MatMul/ReadVariableOp&^lstm_cell_184/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
 : : : 2L
$lstm_cell_184/BiasAdd/ReadVariableOp$lstm_cell_184/BiasAdd/ReadVariableOp2J
#lstm_cell_184/MatMul/ReadVariableOp#lstm_cell_184/MatMul/ReadVariableOp2N
%lstm_cell_184/MatMul_1/ReadVariableOp%lstm_cell_184/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
 
_user_specified_nameinputs
º
È
while_cond_3531919
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3531919___redundant_placeholder05
1while_while_cond_3531919___redundant_placeholder15
1while_while_cond_3531919___redundant_placeholder25
1while_while_cond_3531919___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
Ñ8
Ú
while_body_3532963
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_184_matmul_readvariableop_resource_0:	 I
6while_lstm_cell_184_matmul_1_readvariableop_resource_0:	 D
5while_lstm_cell_184_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_184_matmul_readvariableop_resource:	 G
4while_lstm_cell_184_matmul_1_readvariableop_resource:	 B
3while_lstm_cell_184_biasadd_readvariableop_resource:	¢*while/lstm_cell_184/BiasAdd/ReadVariableOp¢)while/lstm_cell_184/MatMul/ReadVariableOp¢+while/lstm_cell_184/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0
)while/lstm_cell_184/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_184_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype0¼
while/lstm_cell_184/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_184/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
+while/lstm_cell_184/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_184_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0£
while/lstm_cell_184/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_184/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
while/lstm_cell_184/addAddV2$while/lstm_cell_184/MatMul:product:0&while/lstm_cell_184/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*while/lstm_cell_184/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_184_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0ª
while/lstm_cell_184/BiasAddBiasAddwhile/lstm_cell_184/add:z:02while/lstm_cell_184/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
#while/lstm_cell_184/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ò
while/lstm_cell_184/splitSplit,while/lstm_cell_184/split/split_dim:output:0$while/lstm_cell_184/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split|
while/lstm_cell_184/SigmoidSigmoid"while/lstm_cell_184/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
while/lstm_cell_184/Sigmoid_1Sigmoid"while/lstm_cell_184/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_184/mulMul!while/lstm_cell_184/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ v
while/lstm_cell_184/ReluRelu"while/lstm_cell_184/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_184/mul_1Mulwhile/lstm_cell_184/Sigmoid:y:0&while/lstm_cell_184/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_184/add_1AddV2while/lstm_cell_184/mul:z:0while/lstm_cell_184/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
while/lstm_cell_184/Sigmoid_2Sigmoid"while/lstm_cell_184/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ s
while/lstm_cell_184/Relu_1Reluwhile/lstm_cell_184/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_184/mul_2Mul!while/lstm_cell_184/Sigmoid_2:y:0(while/lstm_cell_184/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Æ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_184/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_184/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
while/Identity_5Identitywhile/lstm_cell_184/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ó

while/NoOpNoOp+^while/lstm_cell_184/BiasAdd/ReadVariableOp*^while/lstm_cell_184/MatMul/ReadVariableOp,^while/lstm_cell_184/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_184_biasadd_readvariableop_resource5while_lstm_cell_184_biasadd_readvariableop_resource_0"n
4while_lstm_cell_184_matmul_1_readvariableop_resource6while_lstm_cell_184_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_184_matmul_readvariableop_resource4while_lstm_cell_184_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2X
*while/lstm_cell_184/BiasAdd/ReadVariableOp*while/lstm_cell_184/BiasAdd/ReadVariableOp2V
)while/lstm_cell_184/MatMul/ReadVariableOp)while/lstm_cell_184/MatMul/ReadVariableOp2Z
+while/lstm_cell_184/MatMul_1/ReadVariableOp+while/lstm_cell_184/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 

ì
'rnn_model_21_lstm_45_while_cond_3530758F
Brnn_model_21_lstm_45_while_rnn_model_21_lstm_45_while_loop_counterL
Hrnn_model_21_lstm_45_while_rnn_model_21_lstm_45_while_maximum_iterations*
&rnn_model_21_lstm_45_while_placeholder,
(rnn_model_21_lstm_45_while_placeholder_1,
(rnn_model_21_lstm_45_while_placeholder_2,
(rnn_model_21_lstm_45_while_placeholder_3H
Drnn_model_21_lstm_45_while_less_rnn_model_21_lstm_45_strided_slice_1_
[rnn_model_21_lstm_45_while_rnn_model_21_lstm_45_while_cond_3530758___redundant_placeholder0_
[rnn_model_21_lstm_45_while_rnn_model_21_lstm_45_while_cond_3530758___redundant_placeholder1_
[rnn_model_21_lstm_45_while_rnn_model_21_lstm_45_while_cond_3530758___redundant_placeholder2_
[rnn_model_21_lstm_45_while_rnn_model_21_lstm_45_while_cond_3530758___redundant_placeholder3'
#rnn_model_21_lstm_45_while_identity
¶
rnn_model_21/lstm_45/while/LessLess&rnn_model_21_lstm_45_while_placeholderDrnn_model_21_lstm_45_while_less_rnn_model_21_lstm_45_strided_slice_1*
T0*
_output_shapes
: u
#rnn_model_21/lstm_45/while/IdentityIdentity#rnn_model_21/lstm_45/while/Less:z:0*
T0
*
_output_shapes
: "S
#rnn_model_21_lstm_45_while_identity,rnn_model_21/lstm_45/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
ï
ø
/__inference_lstm_cell_184_layer_call_fn_3536385

inputs
states_0
states_1
unknown:	 
	unknown_0:	 
	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_184_layer_call_and_return_conditional_losses_3531701o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/1
áJ
¢
D__inference_lstm_45_layer_call_and_return_conditional_losses_3534833

inputs?
,lstm_cell_183_matmul_readvariableop_resource:	A
.lstm_cell_183_matmul_1_readvariableop_resource:	 <
-lstm_cell_183_biasadd_readvariableop_resource:	
identity¢$lstm_cell_183/BiasAdd/ReadVariableOp¢#lstm_cell_183/MatMul/ReadVariableOp¢%lstm_cell_183/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
#lstm_cell_183/MatMul/ReadVariableOpReadVariableOp,lstm_cell_183_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_183/MatMulMatMulstrided_slice_2:output:0+lstm_cell_183/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%lstm_cell_183/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_183_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_183/MatMul_1MatMulzeros:output:0-lstm_cell_183/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_183/addAddV2lstm_cell_183/MatMul:product:0 lstm_cell_183/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_183/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_183_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_183/BiasAddBiasAddlstm_cell_183/add:z:0,lstm_cell_183/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
lstm_cell_183/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :à
lstm_cell_183/splitSplit&lstm_cell_183/split/split_dim:output:0lstm_cell_183/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitp
lstm_cell_183/SigmoidSigmoidlstm_cell_183/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
lstm_cell_183/Sigmoid_1Sigmoidlstm_cell_183/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y
lstm_cell_183/mulMullstm_cell_183/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
lstm_cell_183/ReluRelulstm_cell_183/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_183/mul_1Mullstm_cell_183/Sigmoid:y:0 lstm_cell_183/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
lstm_cell_183/add_1AddV2lstm_cell_183/mul:z:0lstm_cell_183/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
lstm_cell_183/Sigmoid_2Sigmoidlstm_cell_183/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ g
lstm_cell_183/Relu_1Relulstm_cell_183/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_183/mul_2Mullstm_cell_183/Sigmoid_2:y:0"lstm_cell_183/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_183_matmul_readvariableop_resource.lstm_cell_183_matmul_1_readvariableop_resource-lstm_cell_183_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3534749*
condR
while_cond_3534748*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 Ã
NoOpNoOp%^lstm_cell_183/BiasAdd/ReadVariableOp$^lstm_cell_183/MatMul/ReadVariableOp&^lstm_cell_183/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : : 2L
$lstm_cell_183/BiasAdd/ReadVariableOp$lstm_cell_183/BiasAdd/ReadVariableOp2J
#lstm_cell_183/MatMul/ReadVariableOp#lstm_cell_183/MatMul/ReadVariableOp2N
%lstm_cell_183/MatMul_1/ReadVariableOp%lstm_cell_183/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
º
È
while_cond_3533127
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3533127___redundant_placeholder05
1while_while_cond_3533127___redundant_placeholder15
1while_while_cond_3533127___redundant_placeholder25
1while_while_cond_3533127___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:

¸
)__inference_lstm_47_layer_call_fn_3535614
inputs_0
unknown:	 
	unknown_0:	 
	unknown_1:	
identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_47_layer_call_and_return_conditional_losses_3532183o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0
º
È
while_cond_3535507
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3535507___redundant_placeholder05
1while_while_cond_3535507___redundant_placeholder15
1while_while_cond_3535507___redundant_placeholder25
1while_while_cond_3535507___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
È	
ö
E__inference_dense_21_layer_call_and_return_conditional_losses_3532661

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
×

J__inference_lstm_cell_184_layer_call_and_return_conditional_losses_3531555

inputs

states
states_11
matmul_readvariableop_resource:	 3
 matmul_1_readvariableop_resource:	 .
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 *
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ N
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates
º
È
while_cond_3535078
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3535078___redundant_placeholder05
1while_while_cond_3535078___redundant_placeholder15
1while_while_cond_3535078___redundant_placeholder25
1while_while_cond_3535078___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
®C
Ú

lstm_47_while_body_3533822,
(lstm_47_while_lstm_47_while_loop_counter2
.lstm_47_while_lstm_47_while_maximum_iterations
lstm_47_while_placeholder
lstm_47_while_placeholder_1
lstm_47_while_placeholder_2
lstm_47_while_placeholder_3+
'lstm_47_while_lstm_47_strided_slice_1_0g
clstm_47_while_tensorarrayv2read_tensorlistgetitem_lstm_47_tensorarrayunstack_tensorlistfromtensor_0O
<lstm_47_while_lstm_cell_185_matmul_readvariableop_resource_0:	 Q
>lstm_47_while_lstm_cell_185_matmul_1_readvariableop_resource_0:	 L
=lstm_47_while_lstm_cell_185_biasadd_readvariableop_resource_0:	
lstm_47_while_identity
lstm_47_while_identity_1
lstm_47_while_identity_2
lstm_47_while_identity_3
lstm_47_while_identity_4
lstm_47_while_identity_5)
%lstm_47_while_lstm_47_strided_slice_1e
alstm_47_while_tensorarrayv2read_tensorlistgetitem_lstm_47_tensorarrayunstack_tensorlistfromtensorM
:lstm_47_while_lstm_cell_185_matmul_readvariableop_resource:	 O
<lstm_47_while_lstm_cell_185_matmul_1_readvariableop_resource:	 J
;lstm_47_while_lstm_cell_185_biasadd_readvariableop_resource:	¢2lstm_47/while/lstm_cell_185/BiasAdd/ReadVariableOp¢1lstm_47/while/lstm_cell_185/MatMul/ReadVariableOp¢3lstm_47/while/lstm_cell_185/MatMul_1/ReadVariableOp
?lstm_47/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Î
1lstm_47/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_47_while_tensorarrayv2read_tensorlistgetitem_lstm_47_tensorarrayunstack_tensorlistfromtensor_0lstm_47_while_placeholderHlstm_47/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0¯
1lstm_47/while/lstm_cell_185/MatMul/ReadVariableOpReadVariableOp<lstm_47_while_lstm_cell_185_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype0Ô
"lstm_47/while/lstm_cell_185/MatMulMatMul8lstm_47/while/TensorArrayV2Read/TensorListGetItem:item:09lstm_47/while/lstm_cell_185/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
3lstm_47/while/lstm_cell_185/MatMul_1/ReadVariableOpReadVariableOp>lstm_47_while_lstm_cell_185_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0»
$lstm_47/while/lstm_cell_185/MatMul_1MatMullstm_47_while_placeholder_2;lstm_47/while/lstm_cell_185/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
lstm_47/while/lstm_cell_185/addAddV2,lstm_47/while/lstm_cell_185/MatMul:product:0.lstm_47/while/lstm_cell_185/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
2lstm_47/while/lstm_cell_185/BiasAdd/ReadVariableOpReadVariableOp=lstm_47_while_lstm_cell_185_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0Â
#lstm_47/while/lstm_cell_185/BiasAddBiasAdd#lstm_47/while/lstm_cell_185/add:z:0:lstm_47/while/lstm_cell_185/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
+lstm_47/while/lstm_cell_185/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
!lstm_47/while/lstm_cell_185/splitSplit4lstm_47/while/lstm_cell_185/split/split_dim:output:0,lstm_47/while/lstm_cell_185/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split
#lstm_47/while/lstm_cell_185/SigmoidSigmoid*lstm_47/while/lstm_cell_185/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
%lstm_47/while/lstm_cell_185/Sigmoid_1Sigmoid*lstm_47/while/lstm_cell_185/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
lstm_47/while/lstm_cell_185/mulMul)lstm_47/while/lstm_cell_185/Sigmoid_1:y:0lstm_47_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 lstm_47/while/lstm_cell_185/ReluRelu*lstm_47/while/lstm_cell_185/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ³
!lstm_47/while/lstm_cell_185/mul_1Mul'lstm_47/while/lstm_cell_185/Sigmoid:y:0.lstm_47/while/lstm_cell_185/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¨
!lstm_47/while/lstm_cell_185/add_1AddV2#lstm_47/while/lstm_cell_185/mul:z:0%lstm_47/while/lstm_cell_185/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
%lstm_47/while/lstm_cell_185/Sigmoid_2Sigmoid*lstm_47/while/lstm_cell_185/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"lstm_47/while/lstm_cell_185/Relu_1Relu%lstm_47/while/lstm_cell_185/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ·
!lstm_47/while/lstm_cell_185/mul_2Mul)lstm_47/while/lstm_cell_185/Sigmoid_2:y:00lstm_47/while/lstm_cell_185/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
8lstm_47/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
2lstm_47/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_47_while_placeholder_1Alstm_47/while/TensorArrayV2Write/TensorListSetItem/index:output:0%lstm_47/while/lstm_cell_185/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒU
lstm_47/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_47/while/addAddV2lstm_47_while_placeholderlstm_47/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_47/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_47/while/add_1AddV2(lstm_47_while_lstm_47_while_loop_counterlstm_47/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_47/while/IdentityIdentitylstm_47/while/add_1:z:0^lstm_47/while/NoOp*
T0*
_output_shapes
: 
lstm_47/while/Identity_1Identity.lstm_47_while_lstm_47_while_maximum_iterations^lstm_47/while/NoOp*
T0*
_output_shapes
: q
lstm_47/while/Identity_2Identitylstm_47/while/add:z:0^lstm_47/while/NoOp*
T0*
_output_shapes
: 
lstm_47/while/Identity_3IdentityBlstm_47/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_47/while/NoOp*
T0*
_output_shapes
: 
lstm_47/while/Identity_4Identity%lstm_47/while/lstm_cell_185/mul_2:z:0^lstm_47/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_47/while/Identity_5Identity%lstm_47/while/lstm_cell_185/add_1:z:0^lstm_47/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ó
lstm_47/while/NoOpNoOp3^lstm_47/while/lstm_cell_185/BiasAdd/ReadVariableOp2^lstm_47/while/lstm_cell_185/MatMul/ReadVariableOp4^lstm_47/while/lstm_cell_185/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_47_while_identitylstm_47/while/Identity:output:0"=
lstm_47_while_identity_1!lstm_47/while/Identity_1:output:0"=
lstm_47_while_identity_2!lstm_47/while/Identity_2:output:0"=
lstm_47_while_identity_3!lstm_47/while/Identity_3:output:0"=
lstm_47_while_identity_4!lstm_47/while/Identity_4:output:0"=
lstm_47_while_identity_5!lstm_47/while/Identity_5:output:0"P
%lstm_47_while_lstm_47_strided_slice_1'lstm_47_while_lstm_47_strided_slice_1_0"|
;lstm_47_while_lstm_cell_185_biasadd_readvariableop_resource=lstm_47_while_lstm_cell_185_biasadd_readvariableop_resource_0"~
<lstm_47_while_lstm_cell_185_matmul_1_readvariableop_resource>lstm_47_while_lstm_cell_185_matmul_1_readvariableop_resource_0"z
:lstm_47_while_lstm_cell_185_matmul_readvariableop_resource<lstm_47_while_lstm_cell_185_matmul_readvariableop_resource_0"È
alstm_47_while_tensorarrayv2read_tensorlistgetitem_lstm_47_tensorarrayunstack_tensorlistfromtensorclstm_47_while_tensorarrayv2read_tensorlistgetitem_lstm_47_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2h
2lstm_47/while/lstm_cell_185/BiasAdd/ReadVariableOp2lstm_47/while/lstm_cell_185/BiasAdd/ReadVariableOp2f
1lstm_47/while/lstm_cell_185/MatMul/ReadVariableOp1lstm_47/while/lstm_cell_185/MatMul/ReadVariableOp2j
3lstm_47/while/lstm_cell_185/MatMul_1/ReadVariableOp3lstm_47/while/lstm_cell_185/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
ô
è
I__inference_rnn_model_21_layer_call_and_return_conditional_losses_3533281
x"
lstm_45_3533253:	"
lstm_45_3533255:	 
lstm_45_3533257:	"
lstm_46_3533260:	 "
lstm_46_3533262:	 
lstm_46_3533264:	"
lstm_47_3533267:	 "
lstm_47_3533269:	 
lstm_47_3533271:	"
dense_21_3533274: 
dense_21_3533276:
identity¢ dense_21/StatefulPartitionedCall¢lstm_45/StatefulPartitionedCall¢lstm_46/StatefulPartitionedCall¢lstm_47/StatefulPartitionedCall
lstm_45/StatefulPartitionedCallStatefulPartitionedCallxlstm_45_3533253lstm_45_3533255lstm_45_3533257*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_45_layer_call_and_return_conditional_losses_3533212¨
lstm_46/StatefulPartitionedCallStatefulPartitionedCall(lstm_45/StatefulPartitionedCall:output:0lstm_46_3533260lstm_46_3533262lstm_46_3533264*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_46_layer_call_and_return_conditional_losses_3533047¤
lstm_47/StatefulPartitionedCallStatefulPartitionedCall(lstm_46/StatefulPartitionedCall:output:0lstm_47_3533267lstm_47_3533269lstm_47_3533271*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_47_layer_call_and_return_conditional_losses_3532882
 dense_21/StatefulPartitionedCallStatefulPartitionedCall(lstm_47/StatefulPartitionedCall:output:0dense_21_3533274dense_21_3533276*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_3532661â
reshape_8/PartitionedCallPartitionedCall)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reshape_8_layer_call_and_return_conditional_losses_3532680u
IdentityIdentity"reshape_8/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿÏ
NoOpNoOp!^dense_21/StatefulPartitionedCall ^lstm_45/StatefulPartitionedCall ^lstm_46/StatefulPartitionedCall ^lstm_47/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : 2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2B
lstm_45/StatefulPartitionedCalllstm_45/StatefulPartitionedCall2B
lstm_46/StatefulPartitionedCalllstm_46/StatefulPartitionedCall2B
lstm_47/StatefulPartitionedCalllstm_47/StatefulPartitionedCall:N J
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


_user_specified_namex
º
È
while_cond_3535985
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3535985___redundant_placeholder05
1while_while_cond_3535985___redundant_placeholder15
1while_while_cond_3535985___redundant_placeholder25
1while_while_cond_3535985___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
K
¤
D__inference_lstm_46_layer_call_and_return_conditional_losses_3535163
inputs_0?
,lstm_cell_184_matmul_readvariableop_resource:	 A
.lstm_cell_184_matmul_1_readvariableop_resource:	 <
-lstm_cell_184_biasadd_readvariableop_resource:	
identity¢$lstm_cell_184/BiasAdd/ReadVariableOp¢#lstm_cell_184/MatMul/ReadVariableOp¢%lstm_cell_184/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask
#lstm_cell_184/MatMul/ReadVariableOpReadVariableOp,lstm_cell_184_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_184/MatMulMatMulstrided_slice_2:output:0+lstm_cell_184/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%lstm_cell_184/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_184_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_184/MatMul_1MatMulzeros:output:0-lstm_cell_184/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_184/addAddV2lstm_cell_184/MatMul:product:0 lstm_cell_184/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_184/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_184_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_184/BiasAddBiasAddlstm_cell_184/add:z:0,lstm_cell_184/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
lstm_cell_184/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :à
lstm_cell_184/splitSplit&lstm_cell_184/split/split_dim:output:0lstm_cell_184/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitp
lstm_cell_184/SigmoidSigmoidlstm_cell_184/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
lstm_cell_184/Sigmoid_1Sigmoidlstm_cell_184/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y
lstm_cell_184/mulMullstm_cell_184/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
lstm_cell_184/ReluRelulstm_cell_184/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_184/mul_1Mullstm_cell_184/Sigmoid:y:0 lstm_cell_184/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
lstm_cell_184/add_1AddV2lstm_cell_184/mul:z:0lstm_cell_184/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
lstm_cell_184/Sigmoid_2Sigmoidlstm_cell_184/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ g
lstm_cell_184/Relu_1Relulstm_cell_184/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_184/mul_2Mullstm_cell_184/Sigmoid_2:y:0"lstm_cell_184/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_184_matmul_readvariableop_resource.lstm_cell_184_matmul_1_readvariableop_resource-lstm_cell_184_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3535079*
condR
while_cond_3535078*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ã
NoOpNoOp%^lstm_cell_184/BiasAdd/ReadVariableOp$^lstm_cell_184/MatMul/ReadVariableOp&^lstm_cell_184/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : 2L
$lstm_cell_184/BiasAdd/ReadVariableOp$lstm_cell_184/BiasAdd/ReadVariableOp2J
#lstm_cell_184/MatMul/ReadVariableOp#lstm_cell_184/MatMul/ReadVariableOp2N
%lstm_cell_184/MatMul_1/ReadVariableOp%lstm_cell_184/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0
çK
¢
D__inference_lstm_47_layer_call_and_return_conditional_losses_3532882

inputs?
,lstm_cell_185_matmul_readvariableop_resource:	 A
.lstm_cell_185_matmul_1_readvariableop_resource:	 <
-lstm_cell_185_biasadd_readvariableop_resource:	
identity¢$lstm_cell_185/BiasAdd/ReadVariableOp¢#lstm_cell_185/MatMul/ReadVariableOp¢%lstm_cell_185/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿ D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask
#lstm_cell_185/MatMul/ReadVariableOpReadVariableOp,lstm_cell_185_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_185/MatMulMatMulstrided_slice_2:output:0+lstm_cell_185/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%lstm_cell_185/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_185_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_185/MatMul_1MatMulzeros:output:0-lstm_cell_185/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_185/addAddV2lstm_cell_185/MatMul:product:0 lstm_cell_185/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_185/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_185_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_185/BiasAddBiasAddlstm_cell_185/add:z:0,lstm_cell_185/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
lstm_cell_185/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :à
lstm_cell_185/splitSplit&lstm_cell_185/split/split_dim:output:0lstm_cell_185/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitp
lstm_cell_185/SigmoidSigmoidlstm_cell_185/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
lstm_cell_185/Sigmoid_1Sigmoidlstm_cell_185/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y
lstm_cell_185/mulMullstm_cell_185/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
lstm_cell_185/ReluRelulstm_cell_185/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_185/mul_1Mullstm_cell_185/Sigmoid:y:0 lstm_cell_185/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
lstm_cell_185/add_1AddV2lstm_cell_185/mul:z:0lstm_cell_185/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
lstm_cell_185/Sigmoid_2Sigmoidlstm_cell_185/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ g
lstm_cell_185/Relu_1Relulstm_cell_185/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_185/mul_2Mullstm_cell_185/Sigmoid_2:y:0"lstm_cell_185/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_185_matmul_readvariableop_resource.lstm_cell_185_matmul_1_readvariableop_resource-lstm_cell_185_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3532797*
condR
while_cond_3532796*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ã
NoOpNoOp%^lstm_cell_185/BiasAdd/ReadVariableOp$^lstm_cell_185/MatMul/ReadVariableOp&^lstm_cell_185/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
 : : : 2L
$lstm_cell_185/BiasAdd/ReadVariableOp$lstm_cell_185/BiasAdd/ReadVariableOp2J
#lstm_cell_185/MatMul/ReadVariableOp#lstm_cell_185/MatMul/ReadVariableOp2N
%lstm_cell_185/MatMul_1/ReadVariableOp%lstm_cell_185/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
 
_user_specified_nameinputs
í9
Ú
while_body_3532797
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_185_matmul_readvariableop_resource_0:	 I
6while_lstm_cell_185_matmul_1_readvariableop_resource_0:	 D
5while_lstm_cell_185_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_185_matmul_readvariableop_resource:	 G
4while_lstm_cell_185_matmul_1_readvariableop_resource:	 B
3while_lstm_cell_185_biasadd_readvariableop_resource:	¢*while/lstm_cell_185/BiasAdd/ReadVariableOp¢)while/lstm_cell_185/MatMul/ReadVariableOp¢+while/lstm_cell_185/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0
)while/lstm_cell_185/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_185_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype0¼
while/lstm_cell_185/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_185/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
+while/lstm_cell_185/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_185_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0£
while/lstm_cell_185/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_185/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
while/lstm_cell_185/addAddV2$while/lstm_cell_185/MatMul:product:0&while/lstm_cell_185/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*while/lstm_cell_185/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_185_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0ª
while/lstm_cell_185/BiasAddBiasAddwhile/lstm_cell_185/add:z:02while/lstm_cell_185/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
#while/lstm_cell_185/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ò
while/lstm_cell_185/splitSplit,while/lstm_cell_185/split/split_dim:output:0$while/lstm_cell_185/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split|
while/lstm_cell_185/SigmoidSigmoid"while/lstm_cell_185/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
while/lstm_cell_185/Sigmoid_1Sigmoid"while/lstm_cell_185/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_185/mulMul!while/lstm_cell_185/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ v
while/lstm_cell_185/ReluRelu"while/lstm_cell_185/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_185/mul_1Mulwhile/lstm_cell_185/Sigmoid:y:0&while/lstm_cell_185/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_185/add_1AddV2while/lstm_cell_185/mul:z:0while/lstm_cell_185/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
while/lstm_cell_185/Sigmoid_2Sigmoid"while/lstm_cell_185/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ s
while/lstm_cell_185/Relu_1Reluwhile/lstm_cell_185/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_185/mul_2Mul!while/lstm_cell_185/Sigmoid_2:y:0(while/lstm_cell_185/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : î
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_185/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_185/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
while/Identity_5Identitywhile/lstm_cell_185/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ó

while/NoOpNoOp+^while/lstm_cell_185/BiasAdd/ReadVariableOp*^while/lstm_cell_185/MatMul/ReadVariableOp,^while/lstm_cell_185/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_185_biasadd_readvariableop_resource5while_lstm_cell_185_biasadd_readvariableop_resource_0"n
4while_lstm_cell_185_matmul_1_readvariableop_resource6while_lstm_cell_185_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_185_matmul_readvariableop_resource4while_lstm_cell_185_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2X
*while/lstm_cell_185/BiasAdd/ReadVariableOp*while/lstm_cell_185/BiasAdd/ReadVariableOp2V
)while/lstm_cell_185/MatMul/ReadVariableOp)while/lstm_cell_185/MatMul/ReadVariableOp2Z
+while/lstm_cell_185/MatMul_1/ReadVariableOp+while/lstm_cell_185/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
áJ
¢
D__inference_lstm_45_layer_call_and_return_conditional_losses_3532341

inputs?
,lstm_cell_183_matmul_readvariableop_resource:	A
.lstm_cell_183_matmul_1_readvariableop_resource:	 <
-lstm_cell_183_biasadd_readvariableop_resource:	
identity¢$lstm_cell_183/BiasAdd/ReadVariableOp¢#lstm_cell_183/MatMul/ReadVariableOp¢%lstm_cell_183/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
#lstm_cell_183/MatMul/ReadVariableOpReadVariableOp,lstm_cell_183_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_183/MatMulMatMulstrided_slice_2:output:0+lstm_cell_183/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%lstm_cell_183/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_183_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_183/MatMul_1MatMulzeros:output:0-lstm_cell_183/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_183/addAddV2lstm_cell_183/MatMul:product:0 lstm_cell_183/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_183/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_183_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_183/BiasAddBiasAddlstm_cell_183/add:z:0,lstm_cell_183/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
lstm_cell_183/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :à
lstm_cell_183/splitSplit&lstm_cell_183/split/split_dim:output:0lstm_cell_183/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitp
lstm_cell_183/SigmoidSigmoidlstm_cell_183/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
lstm_cell_183/Sigmoid_1Sigmoidlstm_cell_183/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y
lstm_cell_183/mulMullstm_cell_183/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
lstm_cell_183/ReluRelulstm_cell_183/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_183/mul_1Mullstm_cell_183/Sigmoid:y:0 lstm_cell_183/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
lstm_cell_183/add_1AddV2lstm_cell_183/mul:z:0lstm_cell_183/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
lstm_cell_183/Sigmoid_2Sigmoidlstm_cell_183/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ g
lstm_cell_183/Relu_1Relulstm_cell_183/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_183/mul_2Mullstm_cell_183/Sigmoid_2:y:0"lstm_cell_183/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_183_matmul_readvariableop_resource.lstm_cell_183_matmul_1_readvariableop_resource-lstm_cell_183_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3532257*
condR
while_cond_3532256*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 Ã
NoOpNoOp%^lstm_cell_183/BiasAdd/ReadVariableOp$^lstm_cell_183/MatMul/ReadVariableOp&^lstm_cell_183/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : : 2L
$lstm_cell_183/BiasAdd/ReadVariableOp$lstm_cell_183/BiasAdd/ReadVariableOp2J
#lstm_cell_183/MatMul/ReadVariableOp#lstm_cell_183/MatMul/ReadVariableOp2N
%lstm_cell_183/MatMul_1/ReadVariableOp%lstm_cell_183/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
º
È
while_cond_3535221
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3535221___redundant_placeholder05
1while_while_cond_3535221___redundant_placeholder15
1while_while_cond_3535221___redundant_placeholder25
1while_while_cond_3535221___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
Ñ8
Ú
while_body_3534892
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_183_matmul_readvariableop_resource_0:	I
6while_lstm_cell_183_matmul_1_readvariableop_resource_0:	 D
5while_lstm_cell_183_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_183_matmul_readvariableop_resource:	G
4while_lstm_cell_183_matmul_1_readvariableop_resource:	 B
3while_lstm_cell_183_biasadd_readvariableop_resource:	¢*while/lstm_cell_183/BiasAdd/ReadVariableOp¢)while/lstm_cell_183/MatMul/ReadVariableOp¢+while/lstm_cell_183/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
)while/lstm_cell_183/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_183_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0¼
while/lstm_cell_183/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_183/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
+while/lstm_cell_183/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_183_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0£
while/lstm_cell_183/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_183/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
while/lstm_cell_183/addAddV2$while/lstm_cell_183/MatMul:product:0&while/lstm_cell_183/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*while/lstm_cell_183/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_183_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0ª
while/lstm_cell_183/BiasAddBiasAddwhile/lstm_cell_183/add:z:02while/lstm_cell_183/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
#while/lstm_cell_183/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ò
while/lstm_cell_183/splitSplit,while/lstm_cell_183/split/split_dim:output:0$while/lstm_cell_183/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split|
while/lstm_cell_183/SigmoidSigmoid"while/lstm_cell_183/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
while/lstm_cell_183/Sigmoid_1Sigmoid"while/lstm_cell_183/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_183/mulMul!while/lstm_cell_183/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ v
while/lstm_cell_183/ReluRelu"while/lstm_cell_183/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_183/mul_1Mulwhile/lstm_cell_183/Sigmoid:y:0&while/lstm_cell_183/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_183/add_1AddV2while/lstm_cell_183/mul:z:0while/lstm_cell_183/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
while/lstm_cell_183/Sigmoid_2Sigmoid"while/lstm_cell_183/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ s
while/lstm_cell_183/Relu_1Reluwhile/lstm_cell_183/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_183/mul_2Mul!while/lstm_cell_183/Sigmoid_2:y:0(while/lstm_cell_183/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Æ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_183/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_183/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
while/Identity_5Identitywhile/lstm_cell_183/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ó

while/NoOpNoOp+^while/lstm_cell_183/BiasAdd/ReadVariableOp*^while/lstm_cell_183/MatMul/ReadVariableOp,^while/lstm_cell_183/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_183_biasadd_readvariableop_resource5while_lstm_cell_183_biasadd_readvariableop_resource_0"n
4while_lstm_cell_183_matmul_1_readvariableop_resource6while_lstm_cell_183_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_183_matmul_readvariableop_resource4while_lstm_cell_183_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2X
*while/lstm_cell_183/BiasAdd/ReadVariableOp*while/lstm_cell_183/BiasAdd/ReadVariableOp2V
)while/lstm_cell_183/MatMul/ReadVariableOp)while/lstm_cell_183/MatMul/ReadVariableOp2Z
+while/lstm_cell_183/MatMul_1/ReadVariableOp+while/lstm_cell_183/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
Ñ8
Ú
while_body_3535365
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_184_matmul_readvariableop_resource_0:	 I
6while_lstm_cell_184_matmul_1_readvariableop_resource_0:	 D
5while_lstm_cell_184_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_184_matmul_readvariableop_resource:	 G
4while_lstm_cell_184_matmul_1_readvariableop_resource:	 B
3while_lstm_cell_184_biasadd_readvariableop_resource:	¢*while/lstm_cell_184/BiasAdd/ReadVariableOp¢)while/lstm_cell_184/MatMul/ReadVariableOp¢+while/lstm_cell_184/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0
)while/lstm_cell_184/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_184_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype0¼
while/lstm_cell_184/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_184/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
+while/lstm_cell_184/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_184_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0£
while/lstm_cell_184/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_184/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
while/lstm_cell_184/addAddV2$while/lstm_cell_184/MatMul:product:0&while/lstm_cell_184/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*while/lstm_cell_184/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_184_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0ª
while/lstm_cell_184/BiasAddBiasAddwhile/lstm_cell_184/add:z:02while/lstm_cell_184/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
#while/lstm_cell_184/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ò
while/lstm_cell_184/splitSplit,while/lstm_cell_184/split/split_dim:output:0$while/lstm_cell_184/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split|
while/lstm_cell_184/SigmoidSigmoid"while/lstm_cell_184/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
while/lstm_cell_184/Sigmoid_1Sigmoid"while/lstm_cell_184/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_184/mulMul!while/lstm_cell_184/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ v
while/lstm_cell_184/ReluRelu"while/lstm_cell_184/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_184/mul_1Mulwhile/lstm_cell_184/Sigmoid:y:0&while/lstm_cell_184/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_184/add_1AddV2while/lstm_cell_184/mul:z:0while/lstm_cell_184/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
while/lstm_cell_184/Sigmoid_2Sigmoid"while/lstm_cell_184/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ s
while/lstm_cell_184/Relu_1Reluwhile/lstm_cell_184/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_184/mul_2Mul!while/lstm_cell_184/Sigmoid_2:y:0(while/lstm_cell_184/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Æ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_184/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_184/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
while/Identity_5Identitywhile/lstm_cell_184/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ó

while/NoOpNoOp+^while/lstm_cell_184/BiasAdd/ReadVariableOp*^while/lstm_cell_184/MatMul/ReadVariableOp,^while/lstm_cell_184/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_184_biasadd_readvariableop_resource5while_lstm_cell_184_biasadd_readvariableop_resource_0"n
4while_lstm_cell_184_matmul_1_readvariableop_resource6while_lstm_cell_184_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_184_matmul_readvariableop_resource4while_lstm_cell_184_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2X
*while/lstm_cell_184/BiasAdd/ReadVariableOp*while/lstm_cell_184/BiasAdd/ReadVariableOp2V
)while/lstm_cell_184/MatMul/ReadVariableOp)while/lstm_cell_184/MatMul/ReadVariableOp2Z
+while/lstm_cell_184/MatMul_1/ReadVariableOp+while/lstm_cell_184/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
º
È
while_cond_3534891
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3534891___redundant_placeholder05
1while_while_cond_3534891___redundant_placeholder15
1while_while_cond_3534891___redundant_placeholder25
1while_while_cond_3534891___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
º
È
while_cond_3532406
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3532406___redundant_placeholder05
1while_while_cond_3532406___redundant_placeholder15
1while_while_cond_3532406___redundant_placeholder25
1while_while_cond_3532406___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
×

J__inference_lstm_cell_184_layer_call_and_return_conditional_losses_3531701

inputs

states
states_11
matmul_readvariableop_resource:	 3
 matmul_1_readvariableop_resource:	 .
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 *
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ N
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates
çK
¢
D__inference_lstm_47_layer_call_and_return_conditional_losses_3536216

inputs?
,lstm_cell_185_matmul_readvariableop_resource:	 A
.lstm_cell_185_matmul_1_readvariableop_resource:	 <
-lstm_cell_185_biasadd_readvariableop_resource:	
identity¢$lstm_cell_185/BiasAdd/ReadVariableOp¢#lstm_cell_185/MatMul/ReadVariableOp¢%lstm_cell_185/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
ÿÿÿÿÿÿÿÿÿ D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask
#lstm_cell_185/MatMul/ReadVariableOpReadVariableOp,lstm_cell_185_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_185/MatMulMatMulstrided_slice_2:output:0+lstm_cell_185/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%lstm_cell_185/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_185_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype0
lstm_cell_185/MatMul_1MatMulzeros:output:0-lstm_cell_185/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_185/addAddV2lstm_cell_185/MatMul:product:0 lstm_cell_185/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_185/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_185_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_185/BiasAddBiasAddlstm_cell_185/add:z:0,lstm_cell_185/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
lstm_cell_185/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :à
lstm_cell_185/splitSplit&lstm_cell_185/split/split_dim:output:0lstm_cell_185/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_splitp
lstm_cell_185/SigmoidSigmoidlstm_cell_185/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
lstm_cell_185/Sigmoid_1Sigmoidlstm_cell_185/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y
lstm_cell_185/mulMullstm_cell_185/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
lstm_cell_185/ReluRelulstm_cell_185/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_185/mul_1Mullstm_cell_185/Sigmoid:y:0 lstm_cell_185/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
lstm_cell_185/add_1AddV2lstm_cell_185/mul:z:0lstm_cell_185/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
lstm_cell_185/Sigmoid_2Sigmoidlstm_cell_185/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ g
lstm_cell_185/Relu_1Relulstm_cell_185/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_cell_185/mul_2Mullstm_cell_185/Sigmoid_2:y:0"lstm_cell_185/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_185_matmul_readvariableop_resource.lstm_cell_185_matmul_1_readvariableop_resource-lstm_cell_185_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3536131*
condR
while_cond_3536130*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ã
NoOpNoOp%^lstm_cell_185/BiasAdd/ReadVariableOp$^lstm_cell_185/MatMul/ReadVariableOp&^lstm_cell_185/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
 : : : 2L
$lstm_cell_185/BiasAdd/ReadVariableOp$lstm_cell_185/BiasAdd/ReadVariableOp2J
#lstm_cell_185/MatMul/ReadVariableOp#lstm_cell_185/MatMul/ReadVariableOp2N
%lstm_cell_185/MatMul_1/ReadVariableOp%lstm_cell_185/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
 
_user_specified_nameinputs
#
ñ
while_body_3531219
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_183_3531243_0:	0
while_lstm_cell_183_3531245_0:	 ,
while_lstm_cell_183_3531247_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_183_3531243:	.
while_lstm_cell_183_3531245:	 *
while_lstm_cell_183_3531247:	¢+while/lstm_cell_183/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¼
+while/lstm_cell_183/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_183_3531243_0while_lstm_cell_183_3531245_0while_lstm_cell_183_3531247_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_183_layer_call_and_return_conditional_losses_3531205Ý
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder4while/lstm_cell_183/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity4while/lstm_cell_183/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/Identity_5Identity4while/lstm_cell_183/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z

while/NoOpNoOp,^while/lstm_cell_183/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"<
while_lstm_cell_183_3531243while_lstm_cell_183_3531243_0"<
while_lstm_cell_183_3531245while_lstm_cell_183_3531245_0"<
while_lstm_cell_183_3531247while_lstm_cell_183_3531247_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2Z
+while/lstm_cell_183/StatefulPartitionedCall+while/lstm_cell_183/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
Ñ8
Ú
while_body_3535079
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_184_matmul_readvariableop_resource_0:	 I
6while_lstm_cell_184_matmul_1_readvariableop_resource_0:	 D
5while_lstm_cell_184_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_184_matmul_readvariableop_resource:	 G
4while_lstm_cell_184_matmul_1_readvariableop_resource:	 B
3while_lstm_cell_184_biasadd_readvariableop_resource:	¢*while/lstm_cell_184/BiasAdd/ReadVariableOp¢)while/lstm_cell_184/MatMul/ReadVariableOp¢+while/lstm_cell_184/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0
)while/lstm_cell_184/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_184_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype0¼
while/lstm_cell_184/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_184/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
+while/lstm_cell_184/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_184_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype0£
while/lstm_cell_184/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_184/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
while/lstm_cell_184/addAddV2$while/lstm_cell_184/MatMul:product:0&while/lstm_cell_184/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*while/lstm_cell_184/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_184_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0ª
while/lstm_cell_184/BiasAddBiasAddwhile/lstm_cell_184/add:z:02while/lstm_cell_184/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
#while/lstm_cell_184/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ò
while/lstm_cell_184/splitSplit,while/lstm_cell_184/split/split_dim:output:0$while/lstm_cell_184/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split|
while/lstm_cell_184/SigmoidSigmoid"while/lstm_cell_184/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
while/lstm_cell_184/Sigmoid_1Sigmoid"while/lstm_cell_184/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_184/mulMul!while/lstm_cell_184/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ v
while/lstm_cell_184/ReluRelu"while/lstm_cell_184/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_184/mul_1Mulwhile/lstm_cell_184/Sigmoid:y:0&while/lstm_cell_184/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_184/add_1AddV2while/lstm_cell_184/mul:z:0while/lstm_cell_184/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
while/lstm_cell_184/Sigmoid_2Sigmoid"while/lstm_cell_184/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ s
while/lstm_cell_184/Relu_1Reluwhile/lstm_cell_184/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/lstm_cell_184/mul_2Mul!while/lstm_cell_184/Sigmoid_2:y:0(while/lstm_cell_184/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Æ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_184/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_184/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
while/Identity_5Identitywhile/lstm_cell_184/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ó

while/NoOpNoOp+^while/lstm_cell_184/BiasAdd/ReadVariableOp*^while/lstm_cell_184/MatMul/ReadVariableOp,^while/lstm_cell_184/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_184_biasadd_readvariableop_resource5while_lstm_cell_184_biasadd_readvariableop_resource_0"n
4while_lstm_cell_184_matmul_1_readvariableop_resource6while_lstm_cell_184_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_184_matmul_readvariableop_resource4while_lstm_cell_184_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : 2X
*while/lstm_cell_184/BiasAdd/ReadVariableOp*while/lstm_cell_184/BiasAdd/ReadVariableOp2V
)while/lstm_cell_184/MatMul/ReadVariableOp)while/lstm_cell_184/MatMul/ReadVariableOp2Z
+while/lstm_cell_184/MatMul_1/ReadVariableOp+while/lstm_cell_184/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 

¶
)__inference_lstm_46_layer_call_fn_3535009

inputs
unknown:	 
	unknown_0:	 
	unknown_1:	
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_46_layer_call_and_return_conditional_losses_3532491s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
 : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
 
_user_specified_nameinputs
®8

D__inference_lstm_46_layer_call_and_return_conditional_losses_3531829

inputs(
lstm_cell_184_3531747:	 (
lstm_cell_184_3531749:	 $
lstm_cell_184_3531751:	
identity¢%lstm_cell_184/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maskþ
%lstm_cell_184/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_184_3531747lstm_cell_184_3531749lstm_cell_184_3531751*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_184_layer_call_and_return_conditional_losses_3531701n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¿
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_184_3531747lstm_cell_184_3531749lstm_cell_184_3531751*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3531760*
condR
while_cond_3531759*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ v
NoOpNoOp&^lstm_cell_184/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : 2N
%lstm_cell_184/StatefulPartitionedCall%lstm_cell_184/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
º
È
while_cond_3532112
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_3532112___redundant_placeholder05
1while_while_cond_3532112___redundant_placeholder15
1while_while_cond_3532112___redundant_placeholder25
1while_while_cond_3532112___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:"µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*³
serving_default
?
input_14
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ
@
output_14
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ÜÓ
ï
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
m_layers
		optimizer


signatures"
_tf_keras_model
n
0
1
2
3
4
5
6
7
8
9
10"
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
8
9
10"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
è
trace_0
trace_1
trace_2
trace_32ý
.__inference_rnn_model_21_layer_call_fn_3532708
.__inference_rnn_model_21_layer_call_fn_3533457
.__inference_rnn_model_21_layer_call_fn_3533484
.__inference_rnn_model_21_layer_call_fn_3533333º
±²­
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 ztrace_0ztrace_1ztrace_2ztrace_3
Ô
trace_0
 trace_1
!trace_2
"trace_32é
I__inference_rnn_model_21_layer_call_and_return_conditional_losses_3533922
I__inference_rnn_model_21_layer_call_and_return_conditional_losses_3534360
I__inference_rnn_model_21_layer_call_and_return_conditional_losses_3533364
I__inference_rnn_model_21_layer_call_and_return_conditional_losses_3533395º
±²­
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 ztrace_0z trace_1z!trace_2z"trace_3
ÍBÊ
"__inference__wrapped_model_3531138input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
C
#0
$1
%2
&3
'4"
trackable_list_wrapper
¯
(iter

)beta_1

*beta_2
	+decay
,learning_ratemËmÌmÍmÎmÏmÐmÑmÒmÓmÔmÕvÖv×vØvÙvÚvÛvÜvÝvÞvßvà"
	optimizer
,
-serving_default"
signature_map
<::	2)rnn_model_21/lstm_45/lstm_cell_183/kernel
F:D	 23rnn_model_21/lstm_45/lstm_cell_183/recurrent_kernel
6:42'rnn_model_21/lstm_45/lstm_cell_183/bias
<::	 2)rnn_model_21/lstm_46/lstm_cell_184/kernel
F:D	 23rnn_model_21/lstm_46/lstm_cell_184/recurrent_kernel
6:42'rnn_model_21/lstm_46/lstm_cell_184/bias
<::	 2)rnn_model_21/lstm_47/lstm_cell_185/kernel
F:D	 23rnn_model_21/lstm_47/lstm_cell_185/recurrent_kernel
6:42'rnn_model_21/lstm_47/lstm_cell_185/bias
.:, 2rnn_model_21/dense_21/kernel
(:&2rnn_model_21/dense_21/bias
 "
trackable_list_wrapper
C
#0
$1
%2
&3
'4"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ûBø
.__inference_rnn_model_21_layer_call_fn_3532708input_1"º
±²­
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
õBò
.__inference_rnn_model_21_layer_call_fn_3533457x"º
±²­
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
õBò
.__inference_rnn_model_21_layer_call_fn_3533484x"º
±²­
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
ûBø
.__inference_rnn_model_21_layer_call_fn_3533333input_1"º
±²­
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
B
I__inference_rnn_model_21_layer_call_and_return_conditional_losses_3533922x"º
±²­
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
B
I__inference_rnn_model_21_layer_call_and_return_conditional_losses_3534360x"º
±²­
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
B
I__inference_rnn_model_21_layer_call_and_return_conditional_losses_3533364input_1"º
±²­
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
B
I__inference_rnn_model_21_layer_call_and_return_conditional_losses_3533395input_1"º
±²­
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
Ú
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6_random_generator
7cell
8
state_spec"
_tf_keras_rnn_layer
Ú
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses
?_random_generator
@cell
A
state_spec"
_tf_keras_rnn_layer
Ú
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses
H_random_generator
Icell
J
state_spec"
_tf_keras_rnn_layer
»
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
¥
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses"
_tf_keras_layer
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ÌBÉ
%__inference_signature_wrapper_3533430input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
N
W	variables
X	keras_api
	Ytotal
	Zcount"
_tf_keras_metric
^
[	variables
\	keras_api
	]total
	^count
_
_fn_kwargs"
_tf_keras_metric
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
¹

`states
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
î
ftrace_0
gtrace_1
htrace_2
itrace_32
)__inference_lstm_45_layer_call_fn_3534371
)__inference_lstm_45_layer_call_fn_3534382
)__inference_lstm_45_layer_call_fn_3534393
)__inference_lstm_45_layer_call_fn_3534404Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zftrace_0zgtrace_1zhtrace_2zitrace_3
Ú
jtrace_0
ktrace_1
ltrace_2
mtrace_32ï
D__inference_lstm_45_layer_call_and_return_conditional_losses_3534547
D__inference_lstm_45_layer_call_and_return_conditional_losses_3534690
D__inference_lstm_45_layer_call_and_return_conditional_losses_3534833
D__inference_lstm_45_layer_call_and_return_conditional_losses_3534976Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zjtrace_0zktrace_1zltrace_2zmtrace_3
"
_generic_user_object
ø
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses
t_random_generator
u
state_size

kernel
recurrent_kernel
bias"
_tf_keras_layer
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
¹

vstates
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
î
|trace_0
}trace_1
~trace_2
trace_32
)__inference_lstm_46_layer_call_fn_3534987
)__inference_lstm_46_layer_call_fn_3534998
)__inference_lstm_46_layer_call_fn_3535009
)__inference_lstm_46_layer_call_fn_3535020Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z|trace_0z}trace_1z~trace_2ztrace_3
â
trace_0
trace_1
trace_2
trace_32ï
D__inference_lstm_46_layer_call_and_return_conditional_losses_3535163
D__inference_lstm_46_layer_call_and_return_conditional_losses_3535306
D__inference_lstm_46_layer_call_and_return_conditional_losses_3535449
D__inference_lstm_46_layer_call_and_return_conditional_losses_3535592Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1ztrace_2ztrace_3
"
_generic_user_object

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator

state_size

kernel
recurrent_kernel
bias"
_tf_keras_layer
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
¿
states
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
ö
trace_0
trace_1
trace_2
trace_32
)__inference_lstm_47_layer_call_fn_3535603
)__inference_lstm_47_layer_call_fn_3535614
)__inference_lstm_47_layer_call_fn_3535625
)__inference_lstm_47_layer_call_fn_3535636Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1ztrace_2ztrace_3
â
trace_0
trace_1
trace_2
trace_32ï
D__inference_lstm_47_layer_call_and_return_conditional_losses_3535781
D__inference_lstm_47_layer_call_and_return_conditional_losses_3535926
D__inference_lstm_47_layer_call_and_return_conditional_losses_3536071
D__inference_lstm_47_layer_call_and_return_conditional_losses_3536216Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1ztrace_2ztrace_3
"
_generic_user_object

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
 _random_generator
¡
state_size

kernel
recurrent_kernel
bias"
_tf_keras_layer
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¢non_trainable_variables
£layers
¤metrics
 ¥layer_regularization_losses
¦layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
ð
§trace_02Ñ
*__inference_dense_21_layer_call_fn_3536225¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z§trace_0

¨trace_02ì
E__inference_dense_21_layer_call_and_return_conditional_losses_3536235¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¨trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
©non_trainable_variables
ªlayers
«metrics
 ¬layer_regularization_losses
­layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
ñ
®trace_02Ò
+__inference_reshape_8_layer_call_fn_3536240¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z®trace_0

¯trace_02í
F__inference_reshape_8_layer_call_and_return_conditional_losses_3536253¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¯trace_0
.
Y0
Z1"
trackable_list_wrapper
-
W	variables"
_generic_user_object
:  (2total
:  (2count
.
]0
^1"
trackable_list_wrapper
-
[	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
70"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
)__inference_lstm_45_layer_call_fn_3534371inputs/0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
)__inference_lstm_45_layer_call_fn_3534382inputs/0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
)__inference_lstm_45_layer_call_fn_3534393inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
)__inference_lstm_45_layer_call_fn_3534404inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¬B©
D__inference_lstm_45_layer_call_and_return_conditional_losses_3534547inputs/0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¬B©
D__inference_lstm_45_layer_call_and_return_conditional_losses_3534690inputs/0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ªB§
D__inference_lstm_45_layer_call_and_return_conditional_losses_3534833inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ªB§
D__inference_lstm_45_layer_call_and_return_conditional_losses_3534976inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
²
°non_trainable_variables
±layers
²metrics
 ³layer_regularization_losses
´layer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
Ý
µtrace_0
¶trace_12¢
/__inference_lstm_cell_183_layer_call_fn_3536270
/__inference_lstm_cell_183_layer_call_fn_3536287½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zµtrace_0z¶trace_1

·trace_0
¸trace_12Ø
J__inference_lstm_cell_183_layer_call_and_return_conditional_losses_3536319
J__inference_lstm_cell_183_layer_call_and_return_conditional_losses_3536351½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z·trace_0z¸trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
@0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
)__inference_lstm_46_layer_call_fn_3534987inputs/0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
)__inference_lstm_46_layer_call_fn_3534998inputs/0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
)__inference_lstm_46_layer_call_fn_3535009inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
)__inference_lstm_46_layer_call_fn_3535020inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¬B©
D__inference_lstm_46_layer_call_and_return_conditional_losses_3535163inputs/0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¬B©
D__inference_lstm_46_layer_call_and_return_conditional_losses_3535306inputs/0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ªB§
D__inference_lstm_46_layer_call_and_return_conditional_losses_3535449inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ªB§
D__inference_lstm_46_layer_call_and_return_conditional_losses_3535592inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¹non_trainable_variables
ºlayers
»metrics
 ¼layer_regularization_losses
½layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ý
¾trace_0
¿trace_12¢
/__inference_lstm_cell_184_layer_call_fn_3536368
/__inference_lstm_cell_184_layer_call_fn_3536385½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¾trace_0z¿trace_1

Àtrace_0
Átrace_12Ø
J__inference_lstm_cell_184_layer_call_and_return_conditional_losses_3536417
J__inference_lstm_cell_184_layer_call_and_return_conditional_losses_3536449½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÀtrace_0zÁtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
I0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
)__inference_lstm_47_layer_call_fn_3535603inputs/0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
)__inference_lstm_47_layer_call_fn_3535614inputs/0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
)__inference_lstm_47_layer_call_fn_3535625inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
)__inference_lstm_47_layer_call_fn_3535636inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¬B©
D__inference_lstm_47_layer_call_and_return_conditional_losses_3535781inputs/0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¬B©
D__inference_lstm_47_layer_call_and_return_conditional_losses_3535926inputs/0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ªB§
D__inference_lstm_47_layer_call_and_return_conditional_losses_3536071inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ªB§
D__inference_lstm_47_layer_call_and_return_conditional_losses_3536216inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ânon_trainable_variables
Ãlayers
Ämetrics
 Ålayer_regularization_losses
Ælayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ý
Çtrace_0
Ètrace_12¢
/__inference_lstm_cell_185_layer_call_fn_3536466
/__inference_lstm_cell_185_layer_call_fn_3536483½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÇtrace_0zÈtrace_1

Étrace_0
Êtrace_12Ø
J__inference_lstm_cell_185_layer_call_and_return_conditional_losses_3536515
J__inference_lstm_cell_185_layer_call_and_return_conditional_losses_3536547½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÉtrace_0zÊtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÞBÛ
*__inference_dense_21_layer_call_fn_3536225inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ùBö
E__inference_dense_21_layer_call_and_return_conditional_losses_3536235inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ßBÜ
+__inference_reshape_8_layer_call_fn_3536240inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
F__inference_reshape_8_layer_call_and_return_conditional_losses_3536253inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
/__inference_lstm_cell_183_layer_call_fn_3536270inputsstates/0states/1"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
/__inference_lstm_cell_183_layer_call_fn_3536287inputsstates/0states/1"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
­Bª
J__inference_lstm_cell_183_layer_call_and_return_conditional_losses_3536319inputsstates/0states/1"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
­Bª
J__inference_lstm_cell_183_layer_call_and_return_conditional_losses_3536351inputsstates/0states/1"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
/__inference_lstm_cell_184_layer_call_fn_3536368inputsstates/0states/1"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
/__inference_lstm_cell_184_layer_call_fn_3536385inputsstates/0states/1"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
­Bª
J__inference_lstm_cell_184_layer_call_and_return_conditional_losses_3536417inputsstates/0states/1"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
­Bª
J__inference_lstm_cell_184_layer_call_and_return_conditional_losses_3536449inputsstates/0states/1"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
/__inference_lstm_cell_185_layer_call_fn_3536466inputsstates/0states/1"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
/__inference_lstm_cell_185_layer_call_fn_3536483inputsstates/0states/1"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
­Bª
J__inference_lstm_cell_185_layer_call_and_return_conditional_losses_3536515inputsstates/0states/1"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
­Bª
J__inference_lstm_cell_185_layer_call_and_return_conditional_losses_3536547inputsstates/0states/1"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
A:?	20Adam/rnn_model_21/lstm_45/lstm_cell_183/kernel/m
K:I	 2:Adam/rnn_model_21/lstm_45/lstm_cell_183/recurrent_kernel/m
;:92.Adam/rnn_model_21/lstm_45/lstm_cell_183/bias/m
A:?	 20Adam/rnn_model_21/lstm_46/lstm_cell_184/kernel/m
K:I	 2:Adam/rnn_model_21/lstm_46/lstm_cell_184/recurrent_kernel/m
;:92.Adam/rnn_model_21/lstm_46/lstm_cell_184/bias/m
A:?	 20Adam/rnn_model_21/lstm_47/lstm_cell_185/kernel/m
K:I	 2:Adam/rnn_model_21/lstm_47/lstm_cell_185/recurrent_kernel/m
;:92.Adam/rnn_model_21/lstm_47/lstm_cell_185/bias/m
3:1 2#Adam/rnn_model_21/dense_21/kernel/m
-:+2!Adam/rnn_model_21/dense_21/bias/m
A:?	20Adam/rnn_model_21/lstm_45/lstm_cell_183/kernel/v
K:I	 2:Adam/rnn_model_21/lstm_45/lstm_cell_183/recurrent_kernel/v
;:92.Adam/rnn_model_21/lstm_45/lstm_cell_183/bias/v
A:?	 20Adam/rnn_model_21/lstm_46/lstm_cell_184/kernel/v
K:I	 2:Adam/rnn_model_21/lstm_46/lstm_cell_184/recurrent_kernel/v
;:92.Adam/rnn_model_21/lstm_46/lstm_cell_184/bias/v
A:?	 20Adam/rnn_model_21/lstm_47/lstm_cell_185/kernel/v
K:I	 2:Adam/rnn_model_21/lstm_47/lstm_cell_185/recurrent_kernel/v
;:92.Adam/rnn_model_21/lstm_47/lstm_cell_185/bias/v
3:1 2#Adam/rnn_model_21/dense_21/kernel/v
-:+2!Adam/rnn_model_21/dense_21/bias/v¢
"__inference__wrapped_model_3531138|4¢1
*¢'
%"
input_1ÿÿÿÿÿÿÿÿÿ

ª "7ª4
2
output_1&#
output_1ÿÿÿÿÿÿÿÿÿ¥
E__inference_dense_21_layer_call_and_return_conditional_losses_3536235\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
*__inference_dense_21_layer_call_fn_3536225O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿÓ
D__inference_lstm_45_layer_call_and_return_conditional_losses_3534547O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Ó
D__inference_lstm_45_layer_call_and_return_conditional_losses_3534690O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ¹
D__inference_lstm_45_layer_call_and_return_conditional_losses_3534833q?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ


 
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
 ¹
D__inference_lstm_45_layer_call_and_return_conditional_losses_3534976q?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ


 
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
 ª
)__inference_lstm_45_layer_call_fn_3534371}O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ª
)__inference_lstm_45_layer_call_fn_3534382}O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
)__inference_lstm_45_layer_call_fn_3534393d?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ


 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
 
)__inference_lstm_45_layer_call_fn_3534404d?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ


 
p

 
ª "ÿÿÿÿÿÿÿÿÿ
 Ó
D__inference_lstm_46_layer_call_and_return_conditional_losses_3535163O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 

 
p 

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Ó
D__inference_lstm_46_layer_call_and_return_conditional_losses_3535306O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 

 
p

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ¹
D__inference_lstm_46_layer_call_and_return_conditional_losses_3535449q?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ
 

 
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
 ¹
D__inference_lstm_46_layer_call_and_return_conditional_losses_3535592q?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ
 

 
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
 ª
)__inference_lstm_46_layer_call_fn_3534987}O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 

 
p 

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ª
)__inference_lstm_46_layer_call_fn_3534998}O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 

 
p

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
)__inference_lstm_46_layer_call_fn_3535009d?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ
 

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
 
)__inference_lstm_46_layer_call_fn_3535020d?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ
 

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ
 Å
D__inference_lstm_47_layer_call_and_return_conditional_losses_3535781}O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 Å
D__inference_lstm_47_layer_call_and_return_conditional_losses_3535926}O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 µ
D__inference_lstm_47_layer_call_and_return_conditional_losses_3536071m?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ
 

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 µ
D__inference_lstm_47_layer_call_and_return_conditional_losses_3536216m?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ
 

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 
)__inference_lstm_47_layer_call_fn_3535603pO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ 
)__inference_lstm_47_layer_call_fn_3535614pO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ 
)__inference_lstm_47_layer_call_fn_3535625`?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ
 

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ 
)__inference_lstm_47_layer_call_fn_3535636`?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ
 

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ Ì
J__inference_lstm_cell_183_layer_call_and_return_conditional_losses_3536319ý¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ 
"
states/1ÿÿÿÿÿÿÿÿÿ 
p 
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ 
EB

0/1/0ÿÿÿÿÿÿÿÿÿ 

0/1/1ÿÿÿÿÿÿÿÿÿ 
 Ì
J__inference_lstm_cell_183_layer_call_and_return_conditional_losses_3536351ý¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ 
"
states/1ÿÿÿÿÿÿÿÿÿ 
p
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ 
EB

0/1/0ÿÿÿÿÿÿÿÿÿ 

0/1/1ÿÿÿÿÿÿÿÿÿ 
 ¡
/__inference_lstm_cell_183_layer_call_fn_3536270í¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ 
"
states/1ÿÿÿÿÿÿÿÿÿ 
p 
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ 
A>

1/0ÿÿÿÿÿÿÿÿÿ 

1/1ÿÿÿÿÿÿÿÿÿ ¡
/__inference_lstm_cell_183_layer_call_fn_3536287í¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ 
"
states/1ÿÿÿÿÿÿÿÿÿ 
p
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ 
A>

1/0ÿÿÿÿÿÿÿÿÿ 

1/1ÿÿÿÿÿÿÿÿÿ Ì
J__inference_lstm_cell_184_layer_call_and_return_conditional_losses_3536417ý¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ 
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ 
"
states/1ÿÿÿÿÿÿÿÿÿ 
p 
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ 
EB

0/1/0ÿÿÿÿÿÿÿÿÿ 

0/1/1ÿÿÿÿÿÿÿÿÿ 
 Ì
J__inference_lstm_cell_184_layer_call_and_return_conditional_losses_3536449ý¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ 
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ 
"
states/1ÿÿÿÿÿÿÿÿÿ 
p
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ 
EB

0/1/0ÿÿÿÿÿÿÿÿÿ 

0/1/1ÿÿÿÿÿÿÿÿÿ 
 ¡
/__inference_lstm_cell_184_layer_call_fn_3536368í¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ 
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ 
"
states/1ÿÿÿÿÿÿÿÿÿ 
p 
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ 
A>

1/0ÿÿÿÿÿÿÿÿÿ 

1/1ÿÿÿÿÿÿÿÿÿ ¡
/__inference_lstm_cell_184_layer_call_fn_3536385í¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ 
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ 
"
states/1ÿÿÿÿÿÿÿÿÿ 
p
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ 
A>

1/0ÿÿÿÿÿÿÿÿÿ 

1/1ÿÿÿÿÿÿÿÿÿ Ì
J__inference_lstm_cell_185_layer_call_and_return_conditional_losses_3536515ý¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ 
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ 
"
states/1ÿÿÿÿÿÿÿÿÿ 
p 
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ 
EB

0/1/0ÿÿÿÿÿÿÿÿÿ 

0/1/1ÿÿÿÿÿÿÿÿÿ 
 Ì
J__inference_lstm_cell_185_layer_call_and_return_conditional_losses_3536547ý¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ 
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ 
"
states/1ÿÿÿÿÿÿÿÿÿ 
p
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ 
EB

0/1/0ÿÿÿÿÿÿÿÿÿ 

0/1/1ÿÿÿÿÿÿÿÿÿ 
 ¡
/__inference_lstm_cell_185_layer_call_fn_3536466í¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ 
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ 
"
states/1ÿÿÿÿÿÿÿÿÿ 
p 
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ 
A>

1/0ÿÿÿÿÿÿÿÿÿ 

1/1ÿÿÿÿÿÿÿÿÿ ¡
/__inference_lstm_cell_185_layer_call_fn_3536483í¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ 
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ 
"
states/1ÿÿÿÿÿÿÿÿÿ 
p
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ 
A>

1/0ÿÿÿÿÿÿÿÿÿ 

1/1ÿÿÿÿÿÿÿÿÿ ¦
F__inference_reshape_8_layer_call_and_return_conditional_losses_3536253\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_reshape_8_layer_call_fn_3536240O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿË
I__inference_rnn_model_21_layer_call_and_return_conditional_losses_3533364~D¢A
*¢'
%"
input_1ÿÿÿÿÿÿÿÿÿ

ª

trainingp ")¢&

0ÿÿÿÿÿÿÿÿÿ
 Ë
I__inference_rnn_model_21_layer_call_and_return_conditional_losses_3533395~D¢A
*¢'
%"
input_1ÿÿÿÿÿÿÿÿÿ

ª

trainingp")¢&

0ÿÿÿÿÿÿÿÿÿ
 Å
I__inference_rnn_model_21_layer_call_and_return_conditional_losses_3533922x>¢;
$¢!

xÿÿÿÿÿÿÿÿÿ

ª

trainingp ")¢&

0ÿÿÿÿÿÿÿÿÿ
 Å
I__inference_rnn_model_21_layer_call_and_return_conditional_losses_3534360x>¢;
$¢!

xÿÿÿÿÿÿÿÿÿ

ª

trainingp")¢&

0ÿÿÿÿÿÿÿÿÿ
 £
.__inference_rnn_model_21_layer_call_fn_3532708qD¢A
*¢'
%"
input_1ÿÿÿÿÿÿÿÿÿ

ª

trainingp "ÿÿÿÿÿÿÿÿÿ£
.__inference_rnn_model_21_layer_call_fn_3533333qD¢A
*¢'
%"
input_1ÿÿÿÿÿÿÿÿÿ

ª

trainingp"ÿÿÿÿÿÿÿÿÿ
.__inference_rnn_model_21_layer_call_fn_3533457k>¢;
$¢!

xÿÿÿÿÿÿÿÿÿ

ª

trainingp "ÿÿÿÿÿÿÿÿÿ
.__inference_rnn_model_21_layer_call_fn_3533484k>¢;
$¢!

xÿÿÿÿÿÿÿÿÿ

ª

trainingp"ÿÿÿÿÿÿÿÿÿ±
%__inference_signature_wrapper_3533430?¢<
¢ 
5ª2
0
input_1%"
input_1ÿÿÿÿÿÿÿÿÿ
"7ª4
2
output_1&#
output_1ÿÿÿÿÿÿÿÿÿ