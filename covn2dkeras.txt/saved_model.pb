З'
З§
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
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
О
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
executor_typestring 
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"serve*2.1.0-dev201910142v1.12.1-15803-gacb32b90ef8ѓ!
|
conv1/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_nameconv1/kernel
u
 conv1/kernel/Read/ReadVariableOpReadVariableOpconv1/kernel*
dtype0*&
_output_shapes
: 
l

conv1/biasVarHandleOp*
shared_name
conv1/bias*
dtype0*
_output_shapes
: *
shape: 
e
conv1/bias/Read/ReadVariableOpReadVariableOp
conv1/bias*
dtype0*
_output_shapes
: 
t
bn_conv1/gammaVarHandleOp*
shared_namebn_conv1/gamma*
dtype0*
_output_shapes
: *
shape: 
m
"bn_conv1/gamma/Read/ReadVariableOpReadVariableOpbn_conv1/gamma*
dtype0*
_output_shapes
: 
r
bn_conv1/betaVarHandleOp*
shared_namebn_conv1/beta*
dtype0*
_output_shapes
: *
shape: 
k
!bn_conv1/beta/Read/ReadVariableOpReadVariableOpbn_conv1/beta*
dtype0*
_output_shapes
: 

bn_conv1/moving_meanVarHandleOp*
dtype0*
_output_shapes
: *
shape: *%
shared_namebn_conv1/moving_mean
y
(bn_conv1/moving_mean/Read/ReadVariableOpReadVariableOpbn_conv1/moving_mean*
dtype0*
_output_shapes
: 

bn_conv1/moving_varianceVarHandleOp*
dtype0*
_output_shapes
: *
shape: *)
shared_namebn_conv1/moving_variance

,bn_conv1/moving_variance/Read/ReadVariableOpReadVariableOpbn_conv1/moving_variance*
dtype0*
_output_shapes
: 
~
conv2d/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:  *
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*
dtype0*&
_output_shapes
:  
n
conv2d/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
dtype0*
_output_shapes
: 

batch_normalization/gammaVarHandleOp*
dtype0*
_output_shapes
: *
shape: **
shared_namebatch_normalization/gamma

-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
dtype0*
_output_shapes
: 

batch_normalization/betaVarHandleOp*)
shared_namebatch_normalization/beta*
dtype0*
_output_shapes
: *
shape: 

,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
dtype0*
_output_shapes
: 

batch_normalization/moving_meanVarHandleOp*0
shared_name!batch_normalization/moving_mean*
dtype0*
_output_shapes
: *
shape: 

3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
dtype0*
_output_shapes
: 

#batch_normalization/moving_varianceVarHandleOp*4
shared_name%#batch_normalization/moving_variance*
dtype0*
_output_shapes
: *
shape: 

7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
dtype0*
_output_shapes
: 

conv2d_1/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:  * 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*
dtype0*&
_output_shapes
:  
r
conv2d_1/biasVarHandleOp*
shared_nameconv2d_1/bias*
dtype0*
_output_shapes
: *
shape: 
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
dtype0*
_output_shapes
: 

batch_normalization_1/gammaVarHandleOp*,
shared_namebatch_normalization_1/gamma*
dtype0*
_output_shapes
: *
shape: 

/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
dtype0*
_output_shapes
: 

batch_normalization_1/betaVarHandleOp*
dtype0*
_output_shapes
: *
shape: *+
shared_namebatch_normalization_1/beta

.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
dtype0*
_output_shapes
: 

!batch_normalization_1/moving_meanVarHandleOp*
dtype0*
_output_shapes
: *
shape: *2
shared_name#!batch_normalization_1/moving_mean

5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
dtype0*
_output_shapes
: 
Ђ
%batch_normalization_1/moving_varianceVarHandleOp*
dtype0*
_output_shapes
: *
shape: *6
shared_name'%batch_normalization_1/moving_variance

9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
dtype0*
_output_shapes
: 

conv2d_3/kernelVarHandleOp* 
shared_nameconv2d_3/kernel*
dtype0*
_output_shapes
: *
shape: @
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*
dtype0*&
_output_shapes
: @
r
conv2d_3/biasVarHandleOp*
shared_nameconv2d_3/bias*
dtype0*
_output_shapes
: *
shape:@
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
dtype0*
_output_shapes
:@

batch_normalization_3/gammaVarHandleOp*,
shared_namebatch_normalization_3/gamma*
dtype0*
_output_shapes
: *
shape:@

/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
dtype0*
_output_shapes
:@

batch_normalization_3/betaVarHandleOp*+
shared_namebatch_normalization_3/beta*
dtype0*
_output_shapes
: *
shape:@

.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
dtype0*
_output_shapes
:@

!batch_normalization_3/moving_meanVarHandleOp*
dtype0*
_output_shapes
: *
shape:@*2
shared_name#!batch_normalization_3/moving_mean

5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
dtype0*
_output_shapes
:@
Ђ
%batch_normalization_3/moving_varianceVarHandleOp*6
shared_name'%batch_normalization_3/moving_variance*
dtype0*
_output_shapes
: *
shape:@

9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
dtype0*
_output_shapes
:@

conv2d_4/kernelVarHandleOp* 
shared_nameconv2d_4/kernel*
dtype0*
_output_shapes
: *
shape:@@
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*
dtype0*&
_output_shapes
:@@
r
conv2d_4/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:@*
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
dtype0*
_output_shapes
:@

batch_normalization_4/gammaVarHandleOp*,
shared_namebatch_normalization_4/gamma*
dtype0*
_output_shapes
: *
shape:@

/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
dtype0*
_output_shapes
:@

batch_normalization_4/betaVarHandleOp*+
shared_namebatch_normalization_4/beta*
dtype0*
_output_shapes
: *
shape:@

.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
dtype0*
_output_shapes
:@

!batch_normalization_4/moving_meanVarHandleOp*
dtype0*
_output_shapes
: *
shape:@*2
shared_name#!batch_normalization_4/moving_mean

5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
dtype0*
_output_shapes
:@
Ђ
%batch_normalization_4/moving_varianceVarHandleOp*6
shared_name'%batch_normalization_4/moving_variance*
dtype0*
_output_shapes
: *
shape:@

9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
dtype0*
_output_shapes
:@

conv2d_2/kernelVarHandleOp* 
shared_nameconv2d_2/kernel*
dtype0*
_output_shapes
: *
shape: @
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*
dtype0*&
_output_shapes
: @
r
conv2d_2/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:@*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
dtype0*
_output_shapes
:@

batch_normalization_2/gammaVarHandleOp*
dtype0*
_output_shapes
: *
shape:@*,
shared_namebatch_normalization_2/gamma

/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
dtype0*
_output_shapes
:@

batch_normalization_2/betaVarHandleOp*+
shared_namebatch_normalization_2/beta*
dtype0*
_output_shapes
: *
shape:@

.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
dtype0*
_output_shapes
:@

!batch_normalization_2/moving_meanVarHandleOp*
dtype0*
_output_shapes
: *
shape:@*2
shared_name#!batch_normalization_2/moving_mean

5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
dtype0*
_output_shapes
:@
Ђ
%batch_normalization_2/moving_varianceVarHandleOp*
dtype0*
_output_shapes
: *
shape:@*6
shared_name'%batch_normalization_2/moving_variance

9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
dtype0*
_output_shapes
:@
n
	fc/kernelVarHandleOp*
shared_name	fc/kernel*
dtype0*
_output_shapes
: *
shape
:@
g
fc/kernel/Read/ReadVariableOpReadVariableOp	fc/kernel*
dtype0*
_output_shapes

:@
f
fc/biasVarHandleOp*
shared_name	fc/bias*
dtype0*
_output_shapes
: *
shape:
_
fc/bias/Read/ReadVariableOpReadVariableOpfc/bias*
dtype0*
_output_shapes
:
f
	Adam/iterVarHandleOp*
shared_name	Adam/iter*
dtype0	*
_output_shapes
: *
shape: 
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
dtype0	*
_output_shapes
: 
j
Adam/beta_1VarHandleOp*
shared_nameAdam/beta_1*
dtype0*
_output_shapes
: *
shape: 
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
j
Adam/beta_2VarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
h

Adam/decayVarHandleOp*
shared_name
Adam/decay*
dtype0*
_output_shapes
: *
shape: 
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
dtype0*
_output_shapes
: 
x
Adam/learning_rateVarHandleOp*
dtype0*
_output_shapes
: *
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
dtype0*
_output_shapes
: 
^
totalVarHandleOp*
shared_nametotal*
dtype0*
_output_shapes
: *
shape: 
W
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
^
countVarHandleOp*
shared_namecount*
dtype0*
_output_shapes
: *
shape: 
W
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: 

Adam/conv1/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *
shape: *$
shared_nameAdam/conv1/kernel/m

'Adam/conv1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1/kernel/m*
dtype0*&
_output_shapes
: 
z
Adam/conv1/bias/mVarHandleOp*"
shared_nameAdam/conv1/bias/m*
dtype0*
_output_shapes
: *
shape: 
s
%Adam/conv1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1/bias/m*
dtype0*
_output_shapes
: 

Adam/bn_conv1/gamma/mVarHandleOp*
dtype0*
_output_shapes
: *
shape: *&
shared_nameAdam/bn_conv1/gamma/m
{
)Adam/bn_conv1/gamma/m/Read/ReadVariableOpReadVariableOpAdam/bn_conv1/gamma/m*
dtype0*
_output_shapes
: 

Adam/bn_conv1/beta/mVarHandleOp*
dtype0*
_output_shapes
: *
shape: *%
shared_nameAdam/bn_conv1/beta/m
y
(Adam/bn_conv1/beta/m/Read/ReadVariableOpReadVariableOpAdam/bn_conv1/beta/m*
dtype0*
_output_shapes
: 

Adam/conv2d/kernel/mVarHandleOp*%
shared_nameAdam/conv2d/kernel/m*
dtype0*
_output_shapes
: *
shape:  

(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*
dtype0*&
_output_shapes
:  
|
Adam/conv2d/bias/mVarHandleOp*
dtype0*
_output_shapes
: *
shape: *#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
dtype0*
_output_shapes
: 

 Adam/batch_normalization/gamma/mVarHandleOp*1
shared_name" Adam/batch_normalization/gamma/m*
dtype0*
_output_shapes
: *
shape: 

4Adam/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/m*
dtype0*
_output_shapes
: 

Adam/batch_normalization/beta/mVarHandleOp*0
shared_name!Adam/batch_normalization/beta/m*
dtype0*
_output_shapes
: *
shape: 

3Adam/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/m*
dtype0*
_output_shapes
: 

Adam/conv2d_1/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:  *'
shared_nameAdam/conv2d_1/kernel/m

*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*
dtype0*&
_output_shapes
:  

Adam/conv2d_1/bias/mVarHandleOp*
dtype0*
_output_shapes
: *
shape: *%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
dtype0*
_output_shapes
: 

"Adam/batch_normalization_1/gamma/mVarHandleOp*3
shared_name$"Adam/batch_normalization_1/gamma/m*
dtype0*
_output_shapes
: *
shape: 

6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/m*
dtype0*
_output_shapes
: 

!Adam/batch_normalization_1/beta/mVarHandleOp*2
shared_name#!Adam/batch_normalization_1/beta/m*
dtype0*
_output_shapes
: *
shape: 

5Adam/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/m*
dtype0*
_output_shapes
: 

Adam/conv2d_3/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *
shape: @*'
shared_nameAdam/conv2d_3/kernel/m

*Adam/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/m*
dtype0*&
_output_shapes
: @

Adam/conv2d_3/bias/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:@*%
shared_nameAdam/conv2d_3/bias/m
y
(Adam/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/m*
dtype0*
_output_shapes
:@

"Adam/batch_normalization_3/gamma/mVarHandleOp*3
shared_name$"Adam/batch_normalization_3/gamma/m*
dtype0*
_output_shapes
: *
shape:@

6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/m*
dtype0*
_output_shapes
:@

!Adam/batch_normalization_3/beta/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:@*2
shared_name#!Adam/batch_normalization_3/beta/m

5Adam/batch_normalization_3/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/m*
dtype0*
_output_shapes
:@

Adam/conv2d_4/kernel/mVarHandleOp*'
shared_nameAdam/conv2d_4/kernel/m*
dtype0*
_output_shapes
: *
shape:@@

*Adam/conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/m*
dtype0*&
_output_shapes
:@@

Adam/conv2d_4/bias/mVarHandleOp*%
shared_nameAdam/conv2d_4/bias/m*
dtype0*
_output_shapes
: *
shape:@
y
(Adam/conv2d_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/m*
dtype0*
_output_shapes
:@

"Adam/batch_normalization_4/gamma/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:@*3
shared_name$"Adam/batch_normalization_4/gamma/m

6Adam/batch_normalization_4/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_4/gamma/m*
dtype0*
_output_shapes
:@

!Adam/batch_normalization_4/beta/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:@*2
shared_name#!Adam/batch_normalization_4/beta/m

5Adam/batch_normalization_4/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_4/beta/m*
dtype0*
_output_shapes
:@

Adam/conv2d_2/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *
shape: @*'
shared_nameAdam/conv2d_2/kernel/m

*Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/m*
dtype0*&
_output_shapes
: @

Adam/conv2d_2/bias/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:@*%
shared_nameAdam/conv2d_2/bias/m
y
(Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/m*
dtype0*
_output_shapes
:@

"Adam/batch_normalization_2/gamma/mVarHandleOp*3
shared_name$"Adam/batch_normalization_2/gamma/m*
dtype0*
_output_shapes
: *
shape:@

6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/m*
dtype0*
_output_shapes
:@

!Adam/batch_normalization_2/beta/mVarHandleOp*2
shared_name#!Adam/batch_normalization_2/beta/m*
dtype0*
_output_shapes
: *
shape:@

5Adam/batch_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/m*
dtype0*
_output_shapes
:@
|
Adam/fc/kernel/mVarHandleOp*!
shared_nameAdam/fc/kernel/m*
dtype0*
_output_shapes
: *
shape
:@
u
$Adam/fc/kernel/m/Read/ReadVariableOpReadVariableOpAdam/fc/kernel/m*
dtype0*
_output_shapes

:@
t
Adam/fc/bias/mVarHandleOp*
shared_nameAdam/fc/bias/m*
dtype0*
_output_shapes
: *
shape:
m
"Adam/fc/bias/m/Read/ReadVariableOpReadVariableOpAdam/fc/bias/m*
dtype0*
_output_shapes
:

Adam/conv1/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape: *$
shared_nameAdam/conv1/kernel/v

'Adam/conv1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1/kernel/v*
dtype0*&
_output_shapes
: 
z
Adam/conv1/bias/vVarHandleOp*
dtype0*
_output_shapes
: *
shape: *"
shared_nameAdam/conv1/bias/v
s
%Adam/conv1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1/bias/v*
dtype0*
_output_shapes
: 

Adam/bn_conv1/gamma/vVarHandleOp*
dtype0*
_output_shapes
: *
shape: *&
shared_nameAdam/bn_conv1/gamma/v
{
)Adam/bn_conv1/gamma/v/Read/ReadVariableOpReadVariableOpAdam/bn_conv1/gamma/v*
dtype0*
_output_shapes
: 

Adam/bn_conv1/beta/vVarHandleOp*%
shared_nameAdam/bn_conv1/beta/v*
dtype0*
_output_shapes
: *
shape: 
y
(Adam/bn_conv1/beta/v/Read/ReadVariableOpReadVariableOpAdam/bn_conv1/beta/v*
dtype0*
_output_shapes
: 

Adam/conv2d/kernel/vVarHandleOp*%
shared_nameAdam/conv2d/kernel/v*
dtype0*
_output_shapes
: *
shape:  

(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*
dtype0*&
_output_shapes
:  
|
Adam/conv2d/bias/vVarHandleOp*#
shared_nameAdam/conv2d/bias/v*
dtype0*
_output_shapes
: *
shape: 
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
dtype0*
_output_shapes
: 

 Adam/batch_normalization/gamma/vVarHandleOp*1
shared_name" Adam/batch_normalization/gamma/v*
dtype0*
_output_shapes
: *
shape: 

4Adam/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/v*
dtype0*
_output_shapes
: 

Adam/batch_normalization/beta/vVarHandleOp*0
shared_name!Adam/batch_normalization/beta/v*
dtype0*
_output_shapes
: *
shape: 

3Adam/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/v*
dtype0*
_output_shapes
: 

Adam/conv2d_1/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:  *'
shared_nameAdam/conv2d_1/kernel/v

*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*
dtype0*&
_output_shapes
:  

Adam/conv2d_1/bias/vVarHandleOp*%
shared_nameAdam/conv2d_1/bias/v*
dtype0*
_output_shapes
: *
shape: 
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
dtype0*
_output_shapes
: 

"Adam/batch_normalization_1/gamma/vVarHandleOp*
dtype0*
_output_shapes
: *
shape: *3
shared_name$"Adam/batch_normalization_1/gamma/v

6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/v*
dtype0*
_output_shapes
: 

!Adam/batch_normalization_1/beta/vVarHandleOp*2
shared_name#!Adam/batch_normalization_1/beta/v*
dtype0*
_output_shapes
: *
shape: 

5Adam/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/v*
dtype0*
_output_shapes
: 

Adam/conv2d_3/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape: @*'
shared_nameAdam/conv2d_3/kernel/v

*Adam/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/v*
dtype0*&
_output_shapes
: @

Adam/conv2d_3/bias/vVarHandleOp*%
shared_nameAdam/conv2d_3/bias/v*
dtype0*
_output_shapes
: *
shape:@
y
(Adam/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/v*
dtype0*
_output_shapes
:@

"Adam/batch_normalization_3/gamma/vVarHandleOp*3
shared_name$"Adam/batch_normalization_3/gamma/v*
dtype0*
_output_shapes
: *
shape:@

6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/v*
dtype0*
_output_shapes
:@

!Adam/batch_normalization_3/beta/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:@*2
shared_name#!Adam/batch_normalization_3/beta/v

5Adam/batch_normalization_3/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/v*
dtype0*
_output_shapes
:@

Adam/conv2d_4/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:@@*'
shared_nameAdam/conv2d_4/kernel/v

*Adam/conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/v*
dtype0*&
_output_shapes
:@@

Adam/conv2d_4/bias/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:@*%
shared_nameAdam/conv2d_4/bias/v
y
(Adam/conv2d_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/v*
dtype0*
_output_shapes
:@

"Adam/batch_normalization_4/gamma/vVarHandleOp*3
shared_name$"Adam/batch_normalization_4/gamma/v*
dtype0*
_output_shapes
: *
shape:@

6Adam/batch_normalization_4/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_4/gamma/v*
dtype0*
_output_shapes
:@

!Adam/batch_normalization_4/beta/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:@*2
shared_name#!Adam/batch_normalization_4/beta/v

5Adam/batch_normalization_4/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_4/beta/v*
dtype0*
_output_shapes
:@

Adam/conv2d_2/kernel/vVarHandleOp*
shape: @*'
shared_nameAdam/conv2d_2/kernel/v*
dtype0*
_output_shapes
: 

*Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/v*
dtype0*&
_output_shapes
: @

Adam/conv2d_2/bias/vVarHandleOp*
shape:@*%
shared_nameAdam/conv2d_2/bias/v*
dtype0*
_output_shapes
: 
y
(Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/v*
dtype0*
_output_shapes
:@

"Adam/batch_normalization_2/gamma/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:@*3
shared_name$"Adam/batch_normalization_2/gamma/v

6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/v*
dtype0*
_output_shapes
:@

!Adam/batch_normalization_2/beta/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:@*2
shared_name#!Adam/batch_normalization_2/beta/v

5Adam/batch_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/v*
dtype0*
_output_shapes
:@
|
Adam/fc/kernel/vVarHandleOp*
shape
:@*!
shared_nameAdam/fc/kernel/v*
dtype0*
_output_shapes
: 
u
$Adam/fc/kernel/v/Read/ReadVariableOpReadVariableOpAdam/fc/kernel/v*
dtype0*
_output_shapes

:@
t
Adam/fc/bias/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:*
shared_nameAdam/fc/bias/v
m
"Adam/fc/bias/v/Read/ReadVariableOpReadVariableOpAdam/fc/bias/v*
dtype0*
_output_shapes
:

NoOpNoOp
Њ
ConstConst"/device:CPU:0*КЉ
valueЏЉBЋЉ BЃЉ
Ј
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer-11
layer-12
layer_with_weights-6
layer-13
layer_with_weights-7
layer-14
layer-15
layer_with_weights-8
layer-16
layer_with_weights-9
layer-17
layer_with_weights-10
layer-18
layer-19
layer_with_weights-11
layer-20
layer-21
layer-22
layer-23
layer-24
layer_with_weights-12
layer-25
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api
 
signatures
 
h

!kernel
"bias
#trainable_variables
$	variables
%regularization_losses
&	keras_api

'axis
	(gamma
)beta
*moving_mean
+moving_variance
,trainable_variables
-	variables
.regularization_losses
/	keras_api
R
0trainable_variables
1	variables
2regularization_losses
3	keras_api
R
4trainable_variables
5	variables
6regularization_losses
7	keras_api
h

8kernel
9bias
:trainable_variables
;	variables
<regularization_losses
=	keras_api

>axis
	?gamma
@beta
Amoving_mean
Bmoving_variance
Ctrainable_variables
D	variables
Eregularization_losses
F	keras_api
R
Gtrainable_variables
H	variables
Iregularization_losses
J	keras_api
h

Kkernel
Lbias
Mtrainable_variables
N	variables
Oregularization_losses
P	keras_api

Qaxis
	Rgamma
Sbeta
Tmoving_mean
Umoving_variance
Vtrainable_variables
W	variables
Xregularization_losses
Y	keras_api
R
Ztrainable_variables
[	variables
\regularization_losses
]	keras_api
R
^trainable_variables
_	variables
`regularization_losses
a	keras_api
R
btrainable_variables
c	variables
dregularization_losses
e	keras_api
h

fkernel
gbias
htrainable_variables
i	variables
jregularization_losses
k	keras_api

laxis
	mgamma
nbeta
omoving_mean
pmoving_variance
qtrainable_variables
r	variables
sregularization_losses
t	keras_api
R
utrainable_variables
v	variables
wregularization_losses
x	keras_api
h

ykernel
zbias
{trainable_variables
|	variables
}regularization_losses
~	keras_api

axis

gamma
	beta
moving_mean
moving_variance
trainable_variables
	variables
regularization_losses
	keras_api
n
kernel
	bias
trainable_variables
	variables
regularization_losses
	keras_api
V
trainable_variables
	variables
regularization_losses
	keras_api
 
	axis

gamma
	beta
moving_mean
moving_variance
trainable_variables
	variables
regularization_losses
	keras_api
V
trainable_variables
	variables
regularization_losses
	keras_api
V
trainable_variables
 	variables
Ёregularization_losses
Ђ	keras_api
V
Ѓtrainable_variables
Є	variables
Ѕregularization_losses
І	keras_api
V
Їtrainable_variables
Ј	variables
Љregularization_losses
Њ	keras_api
n
Ћkernel
	Ќbias
­trainable_variables
Ў	variables
Џregularization_losses
А	keras_api
н
	Бiter
Вbeta_1
Гbeta_2

Дdecay
Еlearning_rate!mЊ"mЋ(mЌ)m­8mЎ9mЏ?mА@mБKmВLmГRmДSmЕfmЖgmЗmmИnmЙymКzmЛ	mМ	mН	mО	mП	mР	mС	ЋmТ	ЌmУ!vФ"vХ(vЦ)vЧ8vШ9vЩ?vЪ@vЫKvЬLvЭRvЮSvЯfvаgvбmvвnvгyvдzvе	vж	vз	vи	vй	vк	vл	Ћvм	Ќvн
Ю
!0
"1
(2
)3
84
95
?6
@7
K8
L9
R10
S11
f12
g13
m14
n15
y16
z17
18
19
20
21
22
23
Ћ24
Ќ25
В
!0
"1
(2
)3
*4
+5
86
97
?8
@9
A10
B11
K12
L13
R14
S15
T16
U17
f18
g19
m20
n21
o22
p23
y24
z25
26
27
28
29
30
31
32
33
34
35
Ћ36
Ќ37
 

Жmetrics
trainable_variables
Зlayers
 Иlayer_regularization_losses
	variables
regularization_losses
Йnon_trainable_variables
 
XV
VARIABLE_VALUEconv1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1

!0
"1
 

Кmetrics
#trainable_variables
Лlayers
 Мlayer_regularization_losses
$	variables
%regularization_losses
Нnon_trainable_variables
 
YW
VARIABLE_VALUEbn_conv1/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbn_conv1/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbn_conv1/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEbn_conv1/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

(0
)1

(0
)1
*2
+3
 

Оmetrics
,trainable_variables
Пlayers
 Рlayer_regularization_losses
-	variables
.regularization_losses
Сnon_trainable_variables
 
 
 

Тmetrics
0trainable_variables
Уlayers
 Фlayer_regularization_losses
1	variables
2regularization_losses
Хnon_trainable_variables
 
 
 

Цmetrics
4trainable_variables
Чlayers
 Шlayer_regularization_losses
5	variables
6regularization_losses
Щnon_trainable_variables
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

80
91

80
91
 

Ъmetrics
:trainable_variables
Ыlayers
 Ьlayer_regularization_losses
;	variables
<regularization_losses
Эnon_trainable_variables
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

?0
@1

?0
@1
A2
B3
 

Юmetrics
Ctrainable_variables
Яlayers
 аlayer_regularization_losses
D	variables
Eregularization_losses
бnon_trainable_variables
 
 
 

вmetrics
Gtrainable_variables
гlayers
 дlayer_regularization_losses
H	variables
Iregularization_losses
еnon_trainable_variables
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

K0
L1

K0
L1
 

жmetrics
Mtrainable_variables
зlayers
 иlayer_regularization_losses
N	variables
Oregularization_losses
йnon_trainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

R0
S1

R0
S1
T2
U3
 

кmetrics
Vtrainable_variables
лlayers
 мlayer_regularization_losses
W	variables
Xregularization_losses
нnon_trainable_variables
 
 
 

оmetrics
Ztrainable_variables
пlayers
 рlayer_regularization_losses
[	variables
\regularization_losses
сnon_trainable_variables
 
 
 

тmetrics
^trainable_variables
уlayers
 фlayer_regularization_losses
_	variables
`regularization_losses
хnon_trainable_variables
 
 
 

цmetrics
btrainable_variables
чlayers
 шlayer_regularization_losses
c	variables
dregularization_losses
щnon_trainable_variables
[Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

f0
g1

f0
g1
 

ъmetrics
htrainable_variables
ыlayers
 ьlayer_regularization_losses
i	variables
jregularization_losses
эnon_trainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

m0
n1

m0
n1
o2
p3
 

юmetrics
qtrainable_variables
яlayers
 №layer_regularization_losses
r	variables
sregularization_losses
ёnon_trainable_variables
 
 
 

ђmetrics
utrainable_variables
ѓlayers
 єlayer_regularization_losses
v	variables
wregularization_losses
ѕnon_trainable_variables
[Y
VARIABLE_VALUEconv2d_4/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_4/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

y0
z1

y0
z1
 

іmetrics
{trainable_variables
їlayers
 јlayer_regularization_losses
|	variables
}regularization_losses
љnon_trainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_4/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_4/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_4/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_4/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1
 
0
1
2
3
 
Ё
њmetrics
trainable_variables
ћlayers
 ќlayer_regularization_losses
	variables
regularization_losses
§non_trainable_variables
\Z
VARIABLE_VALUEconv2d_2/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_2/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
Ё
ўmetrics
trainable_variables
џlayers
 layer_regularization_losses
	variables
regularization_losses
non_trainable_variables
 
 
 
Ё
metrics
trainable_variables
layers
 layer_regularization_losses
	variables
regularization_losses
non_trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_2/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_2/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_2/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_2/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1
 
0
1
2
3
 
Ё
metrics
trainable_variables
layers
 layer_regularization_losses
	variables
regularization_losses
non_trainable_variables
 
 
 
Ё
metrics
trainable_variables
layers
 layer_regularization_losses
	variables
regularization_losses
non_trainable_variables
 
 
 
Ё
metrics
trainable_variables
layers
 layer_regularization_losses
 	variables
Ёregularization_losses
non_trainable_variables
 
 
 
Ё
metrics
Ѓtrainable_variables
layers
 layer_regularization_losses
Є	variables
Ѕregularization_losses
non_trainable_variables
 
 
 
Ё
metrics
Їtrainable_variables
layers
 layer_regularization_losses
Ј	variables
Љregularization_losses
non_trainable_variables
VT
VARIABLE_VALUE	fc/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEfc/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

Ћ0
Ќ1

Ћ0
Ќ1
 
Ё
metrics
­trainable_variables
layers
 layer_regularization_losses
Ў	variables
Џregularization_losses
non_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

0
Ц
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
 
Z
*0
+1
A2
B3
T4
U5
o6
p7
8
9
10
11
 
 
 
 
 
 
 

*0
+1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

A0
B1
 
 
 
 
 
 
 
 
 
 
 

T0
U1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

o0
p1
 
 
 
 
 
 
 
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 


total

 count
Ё
_fn_kwargs
Ђtrainable_variables
Ѓ	variables
Єregularization_losses
Ѕ	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
 1
 
Ё
Іmetrics
Ђtrainable_variables
Їlayers
 Јlayer_regularization_losses
Ѓ	variables
Єregularization_losses
Љnon_trainable_variables
 
 
 

0
 1
{y
VARIABLE_VALUEAdam/conv1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/bn_conv1/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/bn_conv1/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/batch_normalization/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/batch_normalization/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_1/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_1/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_3/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_3/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_3/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_3/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_4/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_4/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_4/gamma/mQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_4/beta/mPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_2/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_2/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_2/gamma/mRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_2/beta/mQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/fc/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/fc/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/bn_conv1/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/bn_conv1/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/batch_normalization/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/batch_normalization/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_1/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_1/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_3/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_3/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_3/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_3/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_4/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_4/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_4/gamma/vQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_4/beta/vPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_2/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_2/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_2/gamma/vRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_2/beta/vQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/fc/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/fc/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
: 

serving_default_input_1Placeholder*
dtype0*/
_output_shapes
:џџџџџџџџџ		*$
shape:џџџџџџџџџ		
п	
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv1/kernel
conv1/biasbn_conv1/gammabn_conv1/betabn_conv1/moving_meanbn_conv1/moving_varianceconv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_3/kernelconv2d_3/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_4/kernelconv2d_4/biasconv2d_2/kernelconv2d_2/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variancebatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variance	fc/kernelfc/bias*,
_gradient_op_typePartitionedCallUnused*,
f'R%
#__inference_signature_wrapper_24722*
Tout
2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:џџџџџџџџџ*2
Tin+
)2'
O
saver_filenamePlaceholder*
dtype0*
_output_shapes
: *
shape: 
м$
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename conv1/kernel/Read/ReadVariableOpconv1/bias/Read/ReadVariableOp"bn_conv1/gamma/Read/ReadVariableOp!bn_conv1/beta/Read/ReadVariableOp(bn_conv1/moving_mean/Read/ReadVariableOp,bn_conv1/moving_variance/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOpfc/kernel/Read/ReadVariableOpfc/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp'Adam/conv1/kernel/m/Read/ReadVariableOp%Adam/conv1/bias/m/Read/ReadVariableOp)Adam/bn_conv1/gamma/m/Read/ReadVariableOp(Adam/bn_conv1/beta/m/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp4Adam/batch_normalization/gamma/m/Read/ReadVariableOp3Adam/batch_normalization/beta/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_1/beta/m/Read/ReadVariableOp*Adam/conv2d_3/kernel/m/Read/ReadVariableOp(Adam/conv2d_3/bias/m/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_3/beta/m/Read/ReadVariableOp*Adam/conv2d_4/kernel/m/Read/ReadVariableOp(Adam/conv2d_4/bias/m/Read/ReadVariableOp6Adam/batch_normalization_4/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_4/beta/m/Read/ReadVariableOp*Adam/conv2d_2/kernel/m/Read/ReadVariableOp(Adam/conv2d_2/bias/m/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_2/beta/m/Read/ReadVariableOp$Adam/fc/kernel/m/Read/ReadVariableOp"Adam/fc/bias/m/Read/ReadVariableOp'Adam/conv1/kernel/v/Read/ReadVariableOp%Adam/conv1/bias/v/Read/ReadVariableOp)Adam/bn_conv1/gamma/v/Read/ReadVariableOp(Adam/bn_conv1/beta/v/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp4Adam/batch_normalization/gamma/v/Read/ReadVariableOp3Adam/batch_normalization/beta/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_1/beta/v/Read/ReadVariableOp*Adam/conv2d_3/kernel/v/Read/ReadVariableOp(Adam/conv2d_3/bias/v/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_3/beta/v/Read/ReadVariableOp*Adam/conv2d_4/kernel/v/Read/ReadVariableOp(Adam/conv2d_4/bias/v/Read/ReadVariableOp6Adam/batch_normalization_4/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_4/beta/v/Read/ReadVariableOp*Adam/conv2d_2/kernel/v/Read/ReadVariableOp(Adam/conv2d_2/bias/v/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_2/beta/v/Read/ReadVariableOp$Adam/fc/kernel/v/Read/ReadVariableOp"Adam/fc/bias/v/Read/ReadVariableOpConst*
Tout
2**
config_proto

CPU

GPU 2J 8*n
Ting
e2c	*
_output_shapes
: *,
_gradient_op_typePartitionedCallUnused*'
f"R 
__inference__traced_save_26616
У
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1/kernel
conv1/biasbn_conv1/gammabn_conv1/betabn_conv1/moving_meanbn_conv1/moving_varianceconv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_3/kernelconv2d_3/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_4/kernelconv2d_4/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_varianceconv2d_2/kernelconv2d_2/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variance	fc/kernelfc/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv1/kernel/mAdam/conv1/bias/mAdam/bn_conv1/gamma/mAdam/bn_conv1/beta/mAdam/conv2d/kernel/mAdam/conv2d/bias/m Adam/batch_normalization/gamma/mAdam/batch_normalization/beta/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/m"Adam/batch_normalization_1/gamma/m!Adam/batch_normalization_1/beta/mAdam/conv2d_3/kernel/mAdam/conv2d_3/bias/m"Adam/batch_normalization_3/gamma/m!Adam/batch_normalization_3/beta/mAdam/conv2d_4/kernel/mAdam/conv2d_4/bias/m"Adam/batch_normalization_4/gamma/m!Adam/batch_normalization_4/beta/mAdam/conv2d_2/kernel/mAdam/conv2d_2/bias/m"Adam/batch_normalization_2/gamma/m!Adam/batch_normalization_2/beta/mAdam/fc/kernel/mAdam/fc/bias/mAdam/conv1/kernel/vAdam/conv1/bias/vAdam/bn_conv1/gamma/vAdam/bn_conv1/beta/vAdam/conv2d/kernel/vAdam/conv2d/bias/v Adam/batch_normalization/gamma/vAdam/batch_normalization/beta/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/v"Adam/batch_normalization_1/gamma/v!Adam/batch_normalization_1/beta/vAdam/conv2d_3/kernel/vAdam/conv2d_3/bias/v"Adam/batch_normalization_3/gamma/v!Adam/batch_normalization_3/beta/vAdam/conv2d_4/kernel/vAdam/conv2d_4/bias/v"Adam/batch_normalization_4/gamma/v!Adam/batch_normalization_4/beta/vAdam/conv2d_2/kernel/vAdam/conv2d_2/bias/v"Adam/batch_normalization_2/gamma/v!Adam/batch_normalization_2/beta/vAdam/fc/kernel/vAdam/fc/bias/v*
Tout
2**
config_proto

CPU

GPU 2J 8*
_output_shapes
: *m
Tinf
d2b*,
_gradient_op_typePartitionedCallUnused**
f%R#
!__inference__traced_restore_26919ек

ѓ
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_23482

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: 2
LogicalAnd/x^
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
:@2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
:@2
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
dtype0*
_output_shapes
:@2!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
dtype0*
_output_shapes
:@2#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*
is_training( *
epsilon%o:*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:2
FusedBatchNormV3S
ConstConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2
Constь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::::2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp:& "
 
_user_specified_nameinputs
Н
ц
C__inference_bn_conv1_layer_call_and_return_conditional_losses_25360

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: 2
LogicalAnd/x^
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
: 2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
: 2
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
dtype0*
_output_shapes
: 2!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
dtype0*
_output_shapes
: 2#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
epsilon%o:*K
_output_shapes9
7:џџџџџџџџџ : : : : :*
T0*
U0*
is_training( 2
FusedBatchNormV3S
ConstConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2
Constк
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ ::::2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp:& "
 
_user_specified_nameinputs
Љ$

P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_23149

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_23134
assignmovingavg_1_23141
identityЂ#AssignMovingAvg/AssignSubVariableOpЂAssignMovingAvg/ReadVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpЂ AssignMovingAvg_1/ReadVariableOpЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/x^
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
: 2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
: 2
ReadVariableOp_1Q
ConstConst*
valueB *
dtype0*
_output_shapes
: 2
ConstU
Const_1Const*
valueB *
dtype0*
_output_shapes
: 2	
Const_1
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
T0*
U0*
epsilon%o:2
FusedBatchNormV3W
Const_2Const*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2	
Const_2
AssignMovingAvg/sub/xConst*
valueB
 *  ?*(
_class
loc:@AssignMovingAvg/23134*
dtype0*
_output_shapes
: 2
AssignMovingAvg/sub/xЎ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/23134*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_23134*
dtype0*
_output_shapes
: 2 
AssignMovingAvg/ReadVariableOpЫ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/23134*
_output_shapes
: 2
AssignMovingAvg/sub_1Д
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/23134*
_output_shapes
: 2
AssignMovingAvg/mulџ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_23134AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/23134*
dtype0*
_output_shapes
 2%
#AssignMovingAvg/AssignSubVariableOpЃ
AssignMovingAvg_1/sub/xConst*
valueB
 *  ?**
_class 
loc:@AssignMovingAvg_1/23141*
dtype0*
_output_shapes
: 2
AssignMovingAvg_1/sub/xЖ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/23141*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_23141*
dtype0*
_output_shapes
: 2"
 AssignMovingAvg_1/ReadVariableOpз
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/23141*
_output_shapes
: 2
AssignMovingAvg_1/sub_1О
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/23141*
_output_shapes
: 2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_23141AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/23141*
dtype0*
_output_shapes
 2'
%AssignMovingAvg_1/AssignSubVariableOpИ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::::2$
ReadVariableOp_1ReadVariableOp_12@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:& "
 
_user_specified_nameinputs

ѓ
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_25700

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: 2
LogicalAnd/x^
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
: 2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
: 2
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
dtype0*
_output_shapes
: 2!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
dtype0*
_output_shapes
: 2#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*
is_training( *
epsilon%o:*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :2
FusedBatchNormV3S
ConstConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2
Constь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::::2 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_1:& "
 
_user_specified_nameinputs
І
T
8__inference_global_average_pooling2d_layer_call_fn_23653

inputs
identityФ
PartitionedCallPartitionedCallinputs*,
_gradient_op_typePartitionedCallUnused*\
fWRU
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_23647*
Tout
2**
config_proto

CPU

GPU 2J 8*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
Tin
22
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:& "
 
_user_specified_nameinputs
Є
c
G__inference_activation_3_layer_call_and_return_conditional_losses_23963

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ :& "
 
_user_specified_nameinputs
Є
c
G__inference_activation_2_layer_call_and_return_conditional_losses_25723

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ :& "
 
_user_specified_nameinputs

ѓ
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_25988

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: 2
LogicalAnd/x^
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
:@2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
:@2
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
dtype0*
_output_shapes
:@2!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
dtype0*
_output_shapes
:@2#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*
is_training( *
epsilon%o:*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:2
FusedBatchNormV3S
ConstConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2
Constь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::::2 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_1:& "
 
_user_specified_nameinputs

ё
(__inference_bn_conv1_layer_call_fn_25369

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityЂStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*,
_gradient_op_typePartitionedCallUnused*L
fGRE
C__inference_bn_conv1_layer_call_and_return_conditional_losses_23693*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin	
2*/
_output_shapes
:џџџџџџџџџ 2
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
О
Љ
(__inference_conv2d_3_layer_call_fn_23206

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tout
2**
config_proto

CPU

GPU 2J 8*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
Tin
2*,
_gradient_op_typePartitionedCallUnused*L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_231982
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Љ$

P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_25796

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_25781
assignmovingavg_1_25788
identityЂ#AssignMovingAvg/AssignSubVariableOpЂAssignMovingAvg/ReadVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpЂ AssignMovingAvg_1/ReadVariableOpЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/x^
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
:@2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
:@2
ReadVariableOp_1Q
ConstConst*
valueB *
dtype0*
_output_shapes
: 2
ConstU
Const_1Const*
valueB *
dtype0*
_output_shapes
: 2	
Const_1
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
epsilon%o:*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
T0*
U02
FusedBatchNormV3W
Const_2Const*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2	
Const_2
AssignMovingAvg/sub/xConst*
valueB
 *  ?*(
_class
loc:@AssignMovingAvg/25781*
dtype0*
_output_shapes
: 2
AssignMovingAvg/sub/xЎ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/25781*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_25781*
dtype0*
_output_shapes
:@2 
AssignMovingAvg/ReadVariableOpЫ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/25781*
_output_shapes
:@2
AssignMovingAvg/sub_1Д
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/25781*
_output_shapes
:@2
AssignMovingAvg/mulџ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_25781AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/25781*
dtype0*
_output_shapes
 2%
#AssignMovingAvg/AssignSubVariableOpЃ
AssignMovingAvg_1/sub/xConst*
valueB
 *  ?**
_class 
loc:@AssignMovingAvg_1/25788*
dtype0*
_output_shapes
: 2
AssignMovingAvg_1/sub/xЖ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/25788*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_25788*
dtype0*
_output_shapes
:@2"
 AssignMovingAvg_1/ReadVariableOpз
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/25788*
_output_shapes
:@2
AssignMovingAvg_1/sub_1О
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/25788*
_output_shapes
:@2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_25788AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/25788*
dtype0*
_output_shapes
 2'
%AssignMovingAvg_1/AssignSubVariableOpИ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::::2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
ь
j
@__inference_add_1_layer_call_and_return_conditional_losses_24249

inputs
inputs_1
identity_
addAddV2inputsinputs_1*
T0*/
_output_shapes
:џџџџџџџџџ@2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:џџџџџџџџџ@:џџџџџџџџџ@:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
О
Љ
(__inference_conv2d_2_layer_call_fn_23508

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCallUnused*L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_23500*
Tout
2**
config_proto

CPU

GPU 2J 8*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
Tin
22
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Д
ў
5__inference_batch_normalization_4_layer_call_fn_26071

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityЂStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*,
_gradient_op_typePartitionedCallUnused*Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_24105*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:џџџџџџџџџ@*
Tin	
22
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Р	
ж
=__inference_fc_layer_call_and_return_conditional_losses_24297

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
dtype0*
_output_shapes

:@2
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
dtype0*
_output_shapes
:2
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs


м
C__inference_conv2d_1_layer_call_and_return_conditional_losses_23047

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*
dtype0*&
_output_shapes
:  2
Conv2D/ReadVariableOpЕ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
dtype0*
_output_shapes
: 2
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2	
BiasAddЏ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs
$

C__inference_bn_conv1_layer_call_and_return_conditional_losses_22835

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_22820
assignmovingavg_1_22827
identityЂ#AssignMovingAvg/AssignSubVariableOpЂAssignMovingAvg/ReadVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpЂ AssignMovingAvg_1/ReadVariableOpЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/x^
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
: 2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
: 2
ReadVariableOp_1Q
ConstConst*
valueB *
dtype0*
_output_shapes
: 2
ConstU
Const_1Const*
valueB *
dtype0*
_output_shapes
: 2	
Const_1
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*
epsilon%o:*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :2
FusedBatchNormV3W
Const_2Const*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2	
Const_2
AssignMovingAvg/sub/xConst*
valueB
 *  ?*(
_class
loc:@AssignMovingAvg/22820*
dtype0*
_output_shapes
: 2
AssignMovingAvg/sub/xЎ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/22820*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_22820*
dtype0*
_output_shapes
: 2 
AssignMovingAvg/ReadVariableOpЫ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/22820*
_output_shapes
: 2
AssignMovingAvg/sub_1Д
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/22820*
_output_shapes
: 2
AssignMovingAvg/mulџ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_22820AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/22820*
dtype0*
_output_shapes
 2%
#AssignMovingAvg/AssignSubVariableOpЃ
AssignMovingAvg_1/sub/xConst*
valueB
 *  ?**
_class 
loc:@AssignMovingAvg_1/22827*
dtype0*
_output_shapes
: 2
AssignMovingAvg_1/sub/xЖ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/22827*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_22827*
dtype0*
_output_shapes
: 2"
 AssignMovingAvg_1/ReadVariableOpз
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/22827*
_output_shapes
: 2
AssignMovingAvg_1/sub_1О
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/22827*
_output_shapes
: 2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_22827AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/22827*
dtype0*
_output_shapes
 2'
%AssignMovingAvg_1/AssignSubVariableOpИ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::::2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
ъ
h
>__inference_add_layer_call_and_return_conditional_losses_23949

inputs
inputs_1
identity_
addAddV2inputsinputs_1*
T0*/
_output_shapes
:џџџџџџџџџ 2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:џџџџџџџџџ :џџџџџџџџџ :& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
Є
c
G__inference_activation_4_layer_call_and_return_conditional_losses_24058

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:& "
 
_user_specified_nameinputs
Б
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_22879

inputs
identityЌ
MaxPoolMaxPoolinputs*
ksize
*
paddingSAME*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:& "
 
_user_specified_nameinputs
Ъ
ѓ
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_24029

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: 2
LogicalAnd/x^
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
:@2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
:@2
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
dtype0*
_output_shapes
:@2!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
dtype0*
_output_shapes
:@2#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*
is_training( *
epsilon%o:*K
_output_shapes9
7:џџџџџџџџџ@:@:@:@:@:2
FusedBatchNormV3S
ConstConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2
Constк
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ@::::2 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_1:& "
 
_user_specified_nameinputs
ъ
ў
5__inference_batch_normalization_4_layer_call_fn_25997

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityЂStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*,
_gradient_op_typePartitionedCallUnused*Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_23451*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin	
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ё
H
,__inference_activation_1_layer_call_fn_25558

inputs
identityЗ
PartitionedCallPartitionedCallinputs*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:џџџџџџџџџ *
Tin
2*,
_gradient_op_typePartitionedCallUnused*P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_238402
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ :& "
 
_user_specified_nameinputs
Д
ў
5__inference_batch_normalization_1_layer_call_fn_25644

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityЂStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin	
2*/
_output_shapes
:џџџџџџџџџ *,
_gradient_op_typePartitionedCallUnused*Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_239062
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Ъ
ѓ
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_24127

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: 2
LogicalAnd/x^
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
:@2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
:@2
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
dtype0*
_output_shapes
:@2!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
dtype0*
_output_shapes
:@2#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*
is_training( *
epsilon%o:*K
_output_shapes9
7:џџџџџџџџџ@:@:@:@:@:2
FusedBatchNormV3S
ConstConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2
Constк
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ@::::2 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_1:& "
 
_user_specified_nameinputs
ё
H
,__inference_activation_2_layer_call_fn_25728

inputs
identityЗ
PartitionedCallPartitionedCallinputs**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:џџџџџџџџџ *
Tin
2*,
_gradient_op_typePartitionedCallUnused*P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_23935*
Tout
22
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ :& "
 
_user_specified_nameinputs
ѓ#

P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_24197

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_24182
assignmovingavg_1_24189
identityЂ#AssignMovingAvg/AssignSubVariableOpЂAssignMovingAvg/ReadVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpЂ AssignMovingAvg_1/ReadVariableOpЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/x^
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
:@2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
:@2
ReadVariableOp_1Q
ConstConst*
valueB *
dtype0*
_output_shapes
: 2
ConstU
Const_1Const*
valueB *
dtype0*
_output_shapes
: 2	
Const_1
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
epsilon%o:*K
_output_shapes9
7:џџџџџџџџџ@:@:@:@:@:*
T0*
U02
FusedBatchNormV3W
Const_2Const*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2	
Const_2
AssignMovingAvg/sub/xConst*
valueB
 *  ?*(
_class
loc:@AssignMovingAvg/24182*
dtype0*
_output_shapes
: 2
AssignMovingAvg/sub/xЎ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/24182*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_24182*
dtype0*
_output_shapes
:@2 
AssignMovingAvg/ReadVariableOpЫ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/24182*
_output_shapes
:@2
AssignMovingAvg/sub_1Д
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/24182*
_output_shapes
:@2
AssignMovingAvg/mulџ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_24182AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/24182*
dtype0*
_output_shapes
 2%
#AssignMovingAvg/AssignSubVariableOpЃ
AssignMovingAvg_1/sub/xConst*
valueB
 *  ?**
_class 
loc:@AssignMovingAvg_1/24189*
dtype0*
_output_shapes
: 2
AssignMovingAvg_1/sub/xЖ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/24189*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_24189*
dtype0*
_output_shapes
:@2"
 AssignMovingAvg_1/ReadVariableOpз
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/24189*
_output_shapes
:@2
AssignMovingAvg_1/sub_1О
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/24189*
_output_shapes
:@2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_24189AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/24189*
dtype0*
_output_shapes
 2'
%AssignMovingAvg_1/AssignSubVariableOpІ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_12@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp:& "
 
_user_specified_nameinputs
Љ$

P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_23451

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_23436
assignmovingavg_1_23443
identityЂ#AssignMovingAvg/AssignSubVariableOpЂAssignMovingAvg/ReadVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpЂ AssignMovingAvg_1/ReadVariableOpЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/x^
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
:@2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
:@2
ReadVariableOp_1Q
ConstConst*
valueB *
dtype0*
_output_shapes
: 2
ConstU
Const_1Const*
valueB *
dtype0*
_output_shapes
: 2	
Const_1
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*
epsilon%o:*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:2
FusedBatchNormV3W
Const_2Const*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2	
Const_2
AssignMovingAvg/sub/xConst*
valueB
 *  ?*(
_class
loc:@AssignMovingAvg/23436*
dtype0*
_output_shapes
: 2
AssignMovingAvg/sub/xЎ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/23436*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_23436*
dtype0*
_output_shapes
:@2 
AssignMovingAvg/ReadVariableOpЫ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/23436*
_output_shapes
:@2
AssignMovingAvg/sub_1Д
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/23436*
_output_shapes
:@2
AssignMovingAvg/mulџ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_23436AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/23436*
dtype0*
_output_shapes
 2%
#AssignMovingAvg/AssignSubVariableOpЃ
AssignMovingAvg_1/sub/xConst*
valueB
 *  ?**
_class 
loc:@AssignMovingAvg_1/23443*
dtype0*
_output_shapes
: 2
AssignMovingAvg_1/sub/xЖ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/23443*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_23443*
dtype0*
_output_shapes
:@2"
 AssignMovingAvg_1/ReadVariableOpз
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/23443*
_output_shapes
:@2
AssignMovingAvg_1/sub_1О
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/23443*
_output_shapes
:@2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_23443AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/23443*
dtype0*
_output_shapes
 2'
%AssignMovingAvg_1/AssignSubVariableOpИ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::::2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_12@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:& "
 
_user_specified_nameinputs
ў	
к
A__inference_conv2d_layer_call_and_return_conditional_losses_22896

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*
dtype0*&
_output_shapes
:  2
Conv2D/ReadVariableOpЕ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
T0*
strides
*
paddingSAME2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
dtype0*
_output_shapes
: 2
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2	
BiasAddЏ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
Љ$

P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_26210

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_26195
assignmovingavg_1_26202
identityЂ#AssignMovingAvg/AssignSubVariableOpЂAssignMovingAvg/ReadVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpЂ AssignMovingAvg_1/ReadVariableOpЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/x^
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
:@2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
:@2
ReadVariableOp_1Q
ConstConst*
valueB *
dtype0*
_output_shapes
: 2
ConstU
Const_1Const*
valueB *
dtype0*
_output_shapes
: 2	
Const_1
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
epsilon%o:*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
T0*
U02
FusedBatchNormV3W
Const_2Const*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2	
Const_2
AssignMovingAvg/sub/xConst*
valueB
 *  ?*(
_class
loc:@AssignMovingAvg/26195*
dtype0*
_output_shapes
: 2
AssignMovingAvg/sub/xЎ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/26195*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_26195*
dtype0*
_output_shapes
:@2 
AssignMovingAvg/ReadVariableOpЫ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/26195*
_output_shapes
:@2
AssignMovingAvg/sub_1Д
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/26195*
_output_shapes
:@2
AssignMovingAvg/mulџ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_26195AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/26195*
dtype0*
_output_shapes
 2%
#AssignMovingAvg/AssignSubVariableOpЃ
AssignMovingAvg_1/sub/xConst*
valueB
 *  ?**
_class 
loc:@AssignMovingAvg_1/26202*
dtype0*
_output_shapes
: 2
AssignMovingAvg_1/sub/xЖ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/26202*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_26202*
dtype0*
_output_shapes
:@2"
 AssignMovingAvg_1/ReadVariableOpз
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/26202*
_output_shapes
:@2
AssignMovingAvg_1/sub_1О
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/26202*
_output_shapes
:@2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_26202AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/26202*
dtype0*
_output_shapes
 2'
%AssignMovingAvg_1/AssignSubVariableOpИ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::::2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
Љ$

P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_23602

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_23587
assignmovingavg_1_23594
identityЂ#AssignMovingAvg/AssignSubVariableOpЂAssignMovingAvg/ReadVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpЂ AssignMovingAvg_1/ReadVariableOpЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/x^
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
:@2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
:@2
ReadVariableOp_1Q
ConstConst*
valueB *
dtype0*
_output_shapes
: 2
ConstU
Const_1Const*
valueB *
dtype0*
_output_shapes
: 2	
Const_1
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*
epsilon%o:*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:2
FusedBatchNormV3W
Const_2Const*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2	
Const_2
AssignMovingAvg/sub/xConst*
valueB
 *  ?*(
_class
loc:@AssignMovingAvg/23587*
dtype0*
_output_shapes
: 2
AssignMovingAvg/sub/xЎ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/23587*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_23587*
dtype0*
_output_shapes
:@2 
AssignMovingAvg/ReadVariableOpЫ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/23587*
_output_shapes
:@2
AssignMovingAvg/sub_1Д
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/23587*
_output_shapes
:@2
AssignMovingAvg/mulџ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_23587AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/23587*
dtype0*
_output_shapes
 2%
#AssignMovingAvg/AssignSubVariableOpЃ
AssignMovingAvg_1/sub/xConst*
valueB
 *  ?**
_class 
loc:@AssignMovingAvg_1/23594*
dtype0*
_output_shapes
: 2
AssignMovingAvg_1/sub/xЖ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/23594*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_23594*
dtype0*
_output_shapes
:@2"
 AssignMovingAvg_1/ReadVariableOpз
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/23594*
_output_shapes
:@2
AssignMovingAvg_1/sub_1О
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/23594*
_output_shapes
:@2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_23594AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/23594*
dtype0*
_output_shapes
 2'
%AssignMovingAvg_1/AssignSubVariableOpИ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::::2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
ѓ#

P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_25604

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_25589
assignmovingavg_1_25596
identityЂ#AssignMovingAvg/AssignSubVariableOpЂAssignMovingAvg/ReadVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpЂ AssignMovingAvg_1/ReadVariableOpЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/x^
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
: 2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
: 2
ReadVariableOp_1Q
ConstConst*
valueB *
dtype0*
_output_shapes
: 2
ConstU
Const_1Const*
valueB *
dtype0*
_output_shapes
: 2	
Const_1
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
epsilon%o:*K
_output_shapes9
7:џџџџџџџџџ : : : : :*
T0*
U02
FusedBatchNormV3W
Const_2Const*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2	
Const_2
AssignMovingAvg/sub/xConst*
valueB
 *  ?*(
_class
loc:@AssignMovingAvg/25589*
dtype0*
_output_shapes
: 2
AssignMovingAvg/sub/xЎ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/25589*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_25589*
dtype0*
_output_shapes
: 2 
AssignMovingAvg/ReadVariableOpЫ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/25589*
_output_shapes
: 2
AssignMovingAvg/sub_1Д
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/25589*
_output_shapes
: 2
AssignMovingAvg/mulџ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_25589AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/25589*
dtype0*
_output_shapes
 2%
#AssignMovingAvg/AssignSubVariableOpЃ
AssignMovingAvg_1/sub/xConst*
valueB
 *  ?**
_class 
loc:@AssignMovingAvg_1/25596*
dtype0*
_output_shapes
: 2
AssignMovingAvg_1/sub/xЖ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/25596*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_25596*
dtype0*
_output_shapes
: 2"
 AssignMovingAvg_1/ReadVariableOpз
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/25596*
_output_shapes
: 2
AssignMovingAvg_1/sub_1О
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/25596*
_output_shapes
: 2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_25596AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/25596*
dtype0*
_output_shapes
 2'
%AssignMovingAvg_1/AssignSubVariableOpІ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ ::::2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
ё#

N__inference_batch_normalization_layer_call_and_return_conditional_losses_23789

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_23774
assignmovingavg_1_23781
identityЂ#AssignMovingAvg/AssignSubVariableOpЂAssignMovingAvg/ReadVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpЂ AssignMovingAvg_1/ReadVariableOpЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/x^
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
: 2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
: 2
ReadVariableOp_1Q
ConstConst*
valueB *
dtype0*
_output_shapes
: 2
ConstU
Const_1Const*
valueB *
dtype0*
_output_shapes
: 2	
Const_1
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*
epsilon%o:*K
_output_shapes9
7:џџџџџџџџџ : : : : :2
FusedBatchNormV3W
Const_2Const*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2	
Const_2
AssignMovingAvg/sub/xConst*
valueB
 *  ?*(
_class
loc:@AssignMovingAvg/23774*
dtype0*
_output_shapes
: 2
AssignMovingAvg/sub/xЎ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/23774*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_23774*
dtype0*
_output_shapes
: 2 
AssignMovingAvg/ReadVariableOpЫ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/23774*
_output_shapes
: 2
AssignMovingAvg/sub_1Д
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/23774*
_output_shapes
: 2
AssignMovingAvg/mulџ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_23774AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/23774*
dtype0*
_output_shapes
 2%
#AssignMovingAvg/AssignSubVariableOpЃ
AssignMovingAvg_1/sub/xConst*
valueB
 *  ?**
_class 
loc:@AssignMovingAvg_1/23781*
dtype0*
_output_shapes
: 2
AssignMovingAvg_1/sub/xЖ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/23781*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_23781*
dtype0*
_output_shapes
: 2"
 AssignMovingAvg_1/ReadVariableOpз
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/23781*
_output_shapes
: 2
AssignMovingAvg_1/sub_1О
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/23781*
_output_shapes
: 2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_23781AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/23781*
dtype0*
_output_shapes
 2'
%AssignMovingAvg_1/AssignSubVariableOpІ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_12@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp:& "
 
_user_specified_nameinputs
Ђ
a
E__inference_activation_layer_call_and_return_conditional_losses_25383

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ :& "
 
_user_specified_nameinputs
Є
c
G__inference_activation_1_layer_call_and_return_conditional_losses_23840

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ :& "
 
_user_specified_nameinputs
Ъ
ѓ
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_25626

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: 2
LogicalAnd/x^
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
: 2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
: 2
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
dtype0*
_output_shapes
: 2!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
dtype0*
_output_shapes
: 2#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
epsilon%o:*K
_output_shapes9
7:џџџџџџџџџ : : : : :*
T0*
U0*
is_training( 2
FusedBatchNormV3S
ConstConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2
Constк
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ ::::2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp:& "
 
_user_specified_nameinputs
Х
I
-__inference_max_pooling2d_layer_call_fn_22885

inputs
identityг
PartitionedCallPartitionedCallinputs**
config_proto

CPU

GPU 2J 8*
Tin
2*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*,
_gradient_op_typePartitionedCallUnused*Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_22879*
Tout
22
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:& "
 
_user_specified_nameinputs
ѓ#

P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_23884

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_23869
assignmovingavg_1_23876
identityЂ#AssignMovingAvg/AssignSubVariableOpЂAssignMovingAvg/ReadVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpЂ AssignMovingAvg_1/ReadVariableOpЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/x^
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
: 2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
: 2
ReadVariableOp_1Q
ConstConst*
valueB *
dtype0*
_output_shapes
: 2
ConstU
Const_1Const*
valueB *
dtype0*
_output_shapes
: 2	
Const_1
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*
epsilon%o:*K
_output_shapes9
7:џџџџџџџџџ : : : : :2
FusedBatchNormV3W
Const_2Const*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2	
Const_2
AssignMovingAvg/sub/xConst*
valueB
 *  ?*(
_class
loc:@AssignMovingAvg/23869*
dtype0*
_output_shapes
: 2
AssignMovingAvg/sub/xЎ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/23869*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_23869*
dtype0*
_output_shapes
: 2 
AssignMovingAvg/ReadVariableOpЫ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/23869*
_output_shapes
: 2
AssignMovingAvg/sub_1Д
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/23869*
_output_shapes
: 2
AssignMovingAvg/mulџ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_23869AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/23869*
dtype0*
_output_shapes
 2%
#AssignMovingAvg/AssignSubVariableOpЃ
AssignMovingAvg_1/sub/xConst*
valueB
 *  ?**
_class 
loc:@AssignMovingAvg_1/23876*
dtype0*
_output_shapes
: 2
AssignMovingAvg_1/sub/xЖ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/23876*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_23876*
dtype0*
_output_shapes
: 2"
 AssignMovingAvg_1/ReadVariableOpз
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/23876*
_output_shapes
: 2
AssignMovingAvg_1/sub_1О
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/23876*
_output_shapes
: 2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_23876AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/23876*
dtype0*
_output_shapes
 2'
%AssignMovingAvg_1/AssignSubVariableOpІ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_12@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp:& "
 
_user_specified_nameinputs
Д
ў
5__inference_batch_normalization_2_layer_call_fn_26176

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityЂStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*,
_gradient_op_typePartitionedCallUnused*Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_24219*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:џџџџџџџџџ@*
Tin	
22
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ѓ#

P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_24007

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_23992
assignmovingavg_1_23999
identityЂ#AssignMovingAvg/AssignSubVariableOpЂAssignMovingAvg/ReadVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpЂ AssignMovingAvg_1/ReadVariableOpЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/x^
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
:@2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
:@2
ReadVariableOp_1Q
ConstConst*
valueB *
dtype0*
_output_shapes
: 2
ConstU
Const_1Const*
valueB *
dtype0*
_output_shapes
: 2	
Const_1
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
epsilon%o:*K
_output_shapes9
7:џџџџџџџџџ@:@:@:@:@:*
T0*
U02
FusedBatchNormV3W
Const_2Const*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2	
Const_2
AssignMovingAvg/sub/xConst*
valueB
 *  ?*(
_class
loc:@AssignMovingAvg/23992*
dtype0*
_output_shapes
: 2
AssignMovingAvg/sub/xЎ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/23992*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_23992*
dtype0*
_output_shapes
:@2 
AssignMovingAvg/ReadVariableOpЫ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/23992*
_output_shapes
:@2
AssignMovingAvg/sub_1Д
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/23992*
_output_shapes
:@2
AssignMovingAvg/mulџ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_23992AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/23992*
dtype0*
_output_shapes
 2%
#AssignMovingAvg/AssignSubVariableOpЃ
AssignMovingAvg_1/sub/xConst*
valueB
 *  ?**
_class 
loc:@AssignMovingAvg_1/23999*
dtype0*
_output_shapes
: 2
AssignMovingAvg_1/sub/xЖ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/23999*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_23999*
dtype0*
_output_shapes
:@2"
 AssignMovingAvg_1/ReadVariableOpз
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/23999*
_output_shapes
:@2
AssignMovingAvg_1/sub_1О
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/23999*
_output_shapes
:@2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_23999AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/23999*
dtype0*
_output_shapes
 2'
%AssignMovingAvg_1/AssignSubVariableOpІ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ@::::2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_12@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp:& "
 
_user_specified_nameinputs
Я
C
'__inference_flatten_layer_call_fn_26283

inputs
identityЊ
PartitionedCallPartitionedCallinputs*,
_gradient_op_typePartitionedCallUnused*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_24278*
Tout
2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:џџџџџџџџџ@*
Tin
22
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ@:& "
 
_user_specified_nameinputs
ѓ#

P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_26136

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_26121
assignmovingavg_1_26128
identityЂ#AssignMovingAvg/AssignSubVariableOpЂAssignMovingAvg/ReadVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpЂ AssignMovingAvg_1/ReadVariableOpЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/x^
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
:@2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
:@2
ReadVariableOp_1Q
ConstConst*
valueB *
dtype0*
_output_shapes
: 2
ConstU
Const_1Const*
dtype0*
_output_shapes
: *
valueB 2	
Const_1
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
epsilon%o:*K
_output_shapes9
7:џџџџџџџџџ@:@:@:@:@:*
T0*
U02
FusedBatchNormV3W
Const_2Const*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2	
Const_2
AssignMovingAvg/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?*(
_class
loc:@AssignMovingAvg/261212
AssignMovingAvg/sub/xЎ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/26121*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_26121*
dtype0*
_output_shapes
:@2 
AssignMovingAvg/ReadVariableOpЫ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/26121*
_output_shapes
:@2
AssignMovingAvg/sub_1Д
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
_output_shapes
:@*
T0*(
_class
loc:@AssignMovingAvg/261212
AssignMovingAvg/mulџ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_26121AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/26121*
dtype0*
_output_shapes
 2%
#AssignMovingAvg/AssignSubVariableOpЃ
AssignMovingAvg_1/sub/xConst*
valueB
 *  ?**
_class 
loc:@AssignMovingAvg_1/26128*
dtype0*
_output_shapes
: 2
AssignMovingAvg_1/sub/xЖ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
_output_shapes
: *
T0**
_class 
loc:@AssignMovingAvg_1/261282
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_26128*
dtype0*
_output_shapes
:@2"
 AssignMovingAvg_1/ReadVariableOpз
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/26128*
_output_shapes
:@2
AssignMovingAvg_1/sub_1О
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/26128*
_output_shapes
:@2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_26128AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 **
_class 
loc:@AssignMovingAvg_1/26128*
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpІ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*/
_output_shapes
:џџџџџџџџџ@*
T02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ@::::2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_12@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:& "
 
_user_specified_nameinputs
Н
ц
C__inference_bn_conv1_layer_call_and_return_conditional_losses_23715

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
value	B
 Z *
dtype0
2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
value	B
 Z*
dtype0
2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
: 2
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
dtype0*
_output_shapes
: 2!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
epsilon%o:*K
_output_shapes9
7:џџџџџџџџџ : : : : :*
T0*
U0*
is_training( 2
FusedBatchNormV3S
ConstConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2
Constк
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp:& "
 
_user_specified_nameinputs
ъ
ў
5__inference_batch_normalization_2_layer_call_fn_26250

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityЂStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_23633*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin	
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*,
_gradient_op_typePartitionedCallUnused2
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ъ
ў
5__inference_batch_normalization_1_layer_call_fn_25718

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityЂStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tout
2**
config_proto

CPU

GPU 2J 8*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
Tin	
2*,
_gradient_op_typePartitionedCallUnused*Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_231802
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
T02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ѓ#

P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_24105

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_24090
assignmovingavg_1_24097
identityЂ#AssignMovingAvg/AssignSubVariableOpЂAssignMovingAvg/ReadVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpЂ AssignMovingAvg_1/ReadVariableOpЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/x^
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
:@2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
:@2
ReadVariableOp_1Q
ConstConst*
valueB *
dtype0*
_output_shapes
: 2
ConstU
Const_1Const*
valueB *
dtype0*
_output_shapes
: 2	
Const_1
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*K
_output_shapes9
7:џџџџџџџџџ@:@:@:@:@:*
T0*
U0*
epsilon%o:2
FusedBatchNormV3W
Const_2Const*
dtype0*
_output_shapes
: *
valueB
 *Єp}?2	
Const_2
AssignMovingAvg/sub/xConst*
valueB
 *  ?*(
_class
loc:@AssignMovingAvg/24090*
dtype0*
_output_shapes
: 2
AssignMovingAvg/sub/xЎ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/24090*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_24090*
dtype0*
_output_shapes
:@2 
AssignMovingAvg/ReadVariableOpЫ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/24090*
_output_shapes
:@2
AssignMovingAvg/sub_1Д
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/24090*
_output_shapes
:@2
AssignMovingAvg/mulџ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_24090AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
dtype0*
_output_shapes
 *(
_class
loc:@AssignMovingAvg/240902%
#AssignMovingAvg/AssignSubVariableOpЃ
AssignMovingAvg_1/sub/xConst*
_output_shapes
: *
valueB
 *  ?**
_class 
loc:@AssignMovingAvg_1/24097*
dtype02
AssignMovingAvg_1/sub/xЖ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0**
_class 
loc:@AssignMovingAvg_1/24097*
_output_shapes
: *
T02
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_24097*
dtype0*
_output_shapes
:@2"
 AssignMovingAvg_1/ReadVariableOpз
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
_output_shapes
:@*
T0**
_class 
loc:@AssignMovingAvg_1/240972
AssignMovingAvg_1/sub_1О
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
_output_shapes
:@*
T0**
_class 
loc:@AssignMovingAvg_1/240972
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_24097AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/24097*
dtype0*
_output_shapes
 2'
%AssignMovingAvg_1/AssignSubVariableOpІ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ@::::2$
ReadVariableOp_1ReadVariableOp_12@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:& "
 
_user_specified_nameinputs

ѓ
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_25818

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: 2
LogicalAnd/x^
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
:@2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
:@2
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
dtype0*
_output_shapes
:@2!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
T0*
U0*
is_training( *
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
valueB
 *Єp}?*
dtype02
Constь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp:& "
 
_user_specified_nameinputs

ѓ
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_26232

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: 2
LogicalAnd/x^
LogicalAnd/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
:@2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
:@2
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
dtype0*
_output_shapes
:@2!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
T0*
U0*
is_training( *
epsilon%o:2
FusedBatchNormV3S
ConstConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2
Constь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::::2 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_1:& "
 
_user_specified_nameinputs
Є
c
G__inference_activation_2_layer_call_and_return_conditional_losses_23935

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ :& "
 
_user_specified_nameinputs
ї
е
'__inference_ResNet9_layer_call_fn_25175

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32#
statefulpartitionedcall_args_33#
statefulpartitionedcall_args_34#
statefulpartitionedcall_args_35#
statefulpartitionedcall_args_36#
statefulpartitionedcall_args_37#
statefulpartitionedcall_args_38
identityЂStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32statefulpartitionedcall_args_33statefulpartitionedcall_args_34statefulpartitionedcall_args_35statefulpartitionedcall_args_36statefulpartitionedcall_args_37statefulpartitionedcall_args_38*K
fFRD
B__inference_ResNet9_layer_call_and_return_conditional_losses_24447*
Tout
2**
config_proto

CPU

GPU 2J 8*2
Tin+
)2'*'
_output_shapes
:џџџџџџџџџ*,
_gradient_op_typePartitionedCallUnused2
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:џџџџџџџџџ*
T02

Identity"
identityIdentity:output:0*Ш
_input_shapesЖ
Г:џџџџџџџџџ		::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs

ѓ
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_23180

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: 2
LogicalAnd/x^
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
: 2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
: 2
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
dtype0*
_output_shapes
: 2!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
dtype0*
_output_shapes
: 2#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
T0*
U0*
is_training( *
epsilon%o:2
FusedBatchNormV3S
ConstConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2
Constь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::::2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp:& "
 
_user_specified_nameinputs
ц
6
!__inference__traced_restore_26919
file_prefix!
assignvariableop_conv1_kernel!
assignvariableop_1_conv1_bias%
!assignvariableop_2_bn_conv1_gamma$
 assignvariableop_3_bn_conv1_beta+
'assignvariableop_4_bn_conv1_moving_mean/
+assignvariableop_5_bn_conv1_moving_variance$
 assignvariableop_6_conv2d_kernel"
assignvariableop_7_conv2d_bias0
,assignvariableop_8_batch_normalization_gamma/
+assignvariableop_9_batch_normalization_beta7
3assignvariableop_10_batch_normalization_moving_mean;
7assignvariableop_11_batch_normalization_moving_variance'
#assignvariableop_12_conv2d_1_kernel%
!assignvariableop_13_conv2d_1_bias3
/assignvariableop_14_batch_normalization_1_gamma2
.assignvariableop_15_batch_normalization_1_beta9
5assignvariableop_16_batch_normalization_1_moving_mean=
9assignvariableop_17_batch_normalization_1_moving_variance'
#assignvariableop_18_conv2d_3_kernel%
!assignvariableop_19_conv2d_3_bias3
/assignvariableop_20_batch_normalization_3_gamma2
.assignvariableop_21_batch_normalization_3_beta9
5assignvariableop_22_batch_normalization_3_moving_mean=
9assignvariableop_23_batch_normalization_3_moving_variance'
#assignvariableop_24_conv2d_4_kernel%
!assignvariableop_25_conv2d_4_bias3
/assignvariableop_26_batch_normalization_4_gamma2
.assignvariableop_27_batch_normalization_4_beta9
5assignvariableop_28_batch_normalization_4_moving_mean=
9assignvariableop_29_batch_normalization_4_moving_variance'
#assignvariableop_30_conv2d_2_kernel%
!assignvariableop_31_conv2d_2_bias3
/assignvariableop_32_batch_normalization_2_gamma2
.assignvariableop_33_batch_normalization_2_beta9
5assignvariableop_34_batch_normalization_2_moving_mean=
9assignvariableop_35_batch_normalization_2_moving_variance!
assignvariableop_36_fc_kernel
assignvariableop_37_fc_bias!
assignvariableop_38_adam_iter#
assignvariableop_39_adam_beta_1#
assignvariableop_40_adam_beta_2"
assignvariableop_41_adam_decay*
&assignvariableop_42_adam_learning_rate
assignvariableop_43_total
assignvariableop_44_count+
'assignvariableop_45_adam_conv1_kernel_m)
%assignvariableop_46_adam_conv1_bias_m-
)assignvariableop_47_adam_bn_conv1_gamma_m,
(assignvariableop_48_adam_bn_conv1_beta_m,
(assignvariableop_49_adam_conv2d_kernel_m*
&assignvariableop_50_adam_conv2d_bias_m8
4assignvariableop_51_adam_batch_normalization_gamma_m7
3assignvariableop_52_adam_batch_normalization_beta_m.
*assignvariableop_53_adam_conv2d_1_kernel_m,
(assignvariableop_54_adam_conv2d_1_bias_m:
6assignvariableop_55_adam_batch_normalization_1_gamma_m9
5assignvariableop_56_adam_batch_normalization_1_beta_m.
*assignvariableop_57_adam_conv2d_3_kernel_m,
(assignvariableop_58_adam_conv2d_3_bias_m:
6assignvariableop_59_adam_batch_normalization_3_gamma_m9
5assignvariableop_60_adam_batch_normalization_3_beta_m.
*assignvariableop_61_adam_conv2d_4_kernel_m,
(assignvariableop_62_adam_conv2d_4_bias_m:
6assignvariableop_63_adam_batch_normalization_4_gamma_m9
5assignvariableop_64_adam_batch_normalization_4_beta_m.
*assignvariableop_65_adam_conv2d_2_kernel_m,
(assignvariableop_66_adam_conv2d_2_bias_m:
6assignvariableop_67_adam_batch_normalization_2_gamma_m9
5assignvariableop_68_adam_batch_normalization_2_beta_m(
$assignvariableop_69_adam_fc_kernel_m&
"assignvariableop_70_adam_fc_bias_m+
'assignvariableop_71_adam_conv1_kernel_v)
%assignvariableop_72_adam_conv1_bias_v-
)assignvariableop_73_adam_bn_conv1_gamma_v,
(assignvariableop_74_adam_bn_conv1_beta_v,
(assignvariableop_75_adam_conv2d_kernel_v*
&assignvariableop_76_adam_conv2d_bias_v8
4assignvariableop_77_adam_batch_normalization_gamma_v7
3assignvariableop_78_adam_batch_normalization_beta_v.
*assignvariableop_79_adam_conv2d_1_kernel_v,
(assignvariableop_80_adam_conv2d_1_bias_v:
6assignvariableop_81_adam_batch_normalization_1_gamma_v9
5assignvariableop_82_adam_batch_normalization_1_beta_v.
*assignvariableop_83_adam_conv2d_3_kernel_v,
(assignvariableop_84_adam_conv2d_3_bias_v:
6assignvariableop_85_adam_batch_normalization_3_gamma_v9
5assignvariableop_86_adam_batch_normalization_3_beta_v.
*assignvariableop_87_adam_conv2d_4_kernel_v,
(assignvariableop_88_adam_conv2d_4_bias_v:
6assignvariableop_89_adam_batch_normalization_4_gamma_v9
5assignvariableop_90_adam_batch_normalization_4_beta_v.
*assignvariableop_91_adam_conv2d_2_kernel_v,
(assignvariableop_92_adam_conv2d_2_bias_v:
6assignvariableop_93_adam_batch_normalization_2_gamma_v9
5assignvariableop_94_adam_batch_normalization_2_beta_v(
$assignvariableop_95_adam_fc_kernel_v&
"assignvariableop_96_adam_fc_bias_v
identity_98ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_45ЂAssignVariableOp_46ЂAssignVariableOp_47ЂAssignVariableOp_48ЂAssignVariableOp_49ЂAssignVariableOp_5ЂAssignVariableOp_50ЂAssignVariableOp_51ЂAssignVariableOp_52ЂAssignVariableOp_53ЂAssignVariableOp_54ЂAssignVariableOp_55ЂAssignVariableOp_56ЂAssignVariableOp_57ЂAssignVariableOp_58ЂAssignVariableOp_59ЂAssignVariableOp_6ЂAssignVariableOp_60ЂAssignVariableOp_61ЂAssignVariableOp_62ЂAssignVariableOp_63ЂAssignVariableOp_64ЂAssignVariableOp_65ЂAssignVariableOp_66ЂAssignVariableOp_67ЂAssignVariableOp_68ЂAssignVariableOp_69ЂAssignVariableOp_7ЂAssignVariableOp_70ЂAssignVariableOp_71ЂAssignVariableOp_72ЂAssignVariableOp_73ЂAssignVariableOp_74ЂAssignVariableOp_75ЂAssignVariableOp_76ЂAssignVariableOp_77ЂAssignVariableOp_78ЂAssignVariableOp_79ЂAssignVariableOp_8ЂAssignVariableOp_80ЂAssignVariableOp_81ЂAssignVariableOp_82ЂAssignVariableOp_83ЂAssignVariableOp_84ЂAssignVariableOp_85ЂAssignVariableOp_86ЂAssignVariableOp_87ЂAssignVariableOp_88ЂAssignVariableOp_89ЂAssignVariableOp_9ЂAssignVariableOp_90ЂAssignVariableOp_91ЂAssignVariableOp_92ЂAssignVariableOp_93ЂAssignVariableOp_94ЂAssignVariableOp_95ЂAssignVariableOp_96Ђ	RestoreV2ЂRestoreV2_1Т6
RestoreV2/tensor_namesConst"/device:CPU:0*Ю5
valueФ5BС5aB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:a2
RestoreV2/tensor_namesг
RestoreV2/shape_and_slicesConst"/device:CPU:0*з
valueЭBЪaB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:a2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*o
dtypese
c2a	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_conv1_kernelIdentity:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1_biasIdentity_1:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
_output_shapes
:*
T02

Identity_2
AssignVariableOp_2AssignVariableOp!assignvariableop_2_bn_conv1_gammaIdentity_2:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3
AssignVariableOp_3AssignVariableOp assignvariableop_3_bn_conv1_betaIdentity_3:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
_output_shapes
:*
T02

Identity_4
AssignVariableOp_4AssignVariableOp'assignvariableop_4_bn_conv1_moving_meanIdentity_4:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
_output_shapes
:*
T02

Identity_5Ё
AssignVariableOp_5AssignVariableOp+assignvariableop_5_bn_conv1_moving_varianceIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6
AssignVariableOp_6AssignVariableOp assignvariableop_6_conv2d_kernelIdentity_6:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7
AssignVariableOp_7AssignVariableOpassignvariableop_7_conv2d_biasIdentity_7:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8Ђ
AssignVariableOp_8AssignVariableOp,assignvariableop_8_batch_normalization_gammaIdentity_8:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9Ё
AssignVariableOp_9AssignVariableOp+assignvariableop_9_batch_normalization_betaIdentity_9:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
_output_shapes
:*
T02
Identity_10Ќ
AssignVariableOp_10AssignVariableOp3assignvariableop_10_batch_normalization_moving_meanIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11А
AssignVariableOp_11AssignVariableOp7assignvariableop_11_batch_normalization_moving_varianceIdentity_11:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv2d_1_kernelIdentity_12:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv2d_1_biasIdentity_13:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14Ј
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_1_gammaIdentity_14:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15Ї
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_1_betaIdentity_15:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
_output_shapes
:*
T02
Identity_16Ў
AssignVariableOp_16AssignVariableOp5assignvariableop_16_batch_normalization_1_moving_meanIdentity_16:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
_output_shapes
:*
T02
Identity_17В
AssignVariableOp_17AssignVariableOp9assignvariableop_17_batch_normalization_1_moving_varianceIdentity_17:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
_output_shapes
:*
T02
Identity_18
AssignVariableOp_18AssignVariableOp#assignvariableop_18_conv2d_3_kernelIdentity_18:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19
AssignVariableOp_19AssignVariableOp!assignvariableop_19_conv2d_3_biasIdentity_19:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20Ј
AssignVariableOp_20AssignVariableOp/assignvariableop_20_batch_normalization_3_gammaIdentity_20:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
_output_shapes
:*
T02
Identity_21Ї
AssignVariableOp_21AssignVariableOp.assignvariableop_21_batch_normalization_3_betaIdentity_21:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22Ў
AssignVariableOp_22AssignVariableOp5assignvariableop_22_batch_normalization_3_moving_meanIdentity_22:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23В
AssignVariableOp_23AssignVariableOp9assignvariableop_23_batch_normalization_3_moving_varianceIdentity_23:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24
AssignVariableOp_24AssignVariableOp#assignvariableop_24_conv2d_4_kernelIdentity_24:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25
AssignVariableOp_25AssignVariableOp!assignvariableop_25_conv2d_4_biasIdentity_25:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
_output_shapes
:*
T02
Identity_26Ј
AssignVariableOp_26AssignVariableOp/assignvariableop_26_batch_normalization_4_gammaIdentity_26:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27Ї
AssignVariableOp_27AssignVariableOp.assignvariableop_27_batch_normalization_4_betaIdentity_27:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28Ў
AssignVariableOp_28AssignVariableOp5assignvariableop_28_batch_normalization_4_moving_meanIdentity_28:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29В
AssignVariableOp_29AssignVariableOp9assignvariableop_29_batch_normalization_4_moving_varianceIdentity_29:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30
AssignVariableOp_30AssignVariableOp#assignvariableop_30_conv2d_2_kernelIdentity_30:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
_output_shapes
:*
T02
Identity_31
AssignVariableOp_31AssignVariableOp!assignvariableop_31_conv2d_2_biasIdentity_31:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32Ј
AssignVariableOp_32AssignVariableOp/assignvariableop_32_batch_normalization_2_gammaIdentity_32:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33Ї
AssignVariableOp_33AssignVariableOp.assignvariableop_33_batch_normalization_2_betaIdentity_33:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
_output_shapes
:*
T02
Identity_34Ў
AssignVariableOp_34AssignVariableOp5assignvariableop_34_batch_normalization_2_moving_meanIdentity_34:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35В
AssignVariableOp_35AssignVariableOp9assignvariableop_35_batch_normalization_2_moving_varianceIdentity_35:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
_output_shapes
:*
T02
Identity_36
AssignVariableOp_36AssignVariableOpassignvariableop_36_fc_kernelIdentity_36:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
_output_shapes
:*
T02
Identity_37
AssignVariableOp_37AssignVariableOpassignvariableop_37_fc_biasIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
_output_shapes
:*
T0	2
Identity_38
AssignVariableOp_38AssignVariableOpassignvariableop_38_adam_iterIdentity_38:output:0*
dtype0	*
_output_shapes
 2
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39
AssignVariableOp_39AssignVariableOpassignvariableop_39_adam_beta_1Identity_39:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40
AssignVariableOp_40AssignVariableOpassignvariableop_40_adam_beta_2Identity_40:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41
AssignVariableOp_41AssignVariableOpassignvariableop_41_adam_decayIdentity_41:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42
AssignVariableOp_42AssignVariableOp&assignvariableop_42_adam_learning_rateIdentity_42:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_42_
Identity_43IdentityRestoreV2:tensors:43*
_output_shapes
:*
T02
Identity_43
AssignVariableOp_43AssignVariableOpassignvariableop_43_totalIdentity_43:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:2
Identity_44
AssignVariableOp_44AssignVariableOpassignvariableop_44_countIdentity_44:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_44_
Identity_45IdentityRestoreV2:tensors:45*
_output_shapes
:*
T02
Identity_45 
AssignVariableOp_45AssignVariableOp'assignvariableop_45_adam_conv1_kernel_mIdentity_45:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_45_
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:2
Identity_46
AssignVariableOp_46AssignVariableOp%assignvariableop_46_adam_conv1_bias_mIdentity_46:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_46_
Identity_47IdentityRestoreV2:tensors:47*
_output_shapes
:*
T02
Identity_47Ђ
AssignVariableOp_47AssignVariableOp)assignvariableop_47_adam_bn_conv1_gamma_mIdentity_47:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_47_
Identity_48IdentityRestoreV2:tensors:48*
T0*
_output_shapes
:2
Identity_48Ё
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_bn_conv1_beta_mIdentity_48:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_48_
Identity_49IdentityRestoreV2:tensors:49*
_output_shapes
:*
T02
Identity_49Ё
AssignVariableOp_49AssignVariableOp(assignvariableop_49_adam_conv2d_kernel_mIdentity_49:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_49_
Identity_50IdentityRestoreV2:tensors:50*
T0*
_output_shapes
:2
Identity_50
AssignVariableOp_50AssignVariableOp&assignvariableop_50_adam_conv2d_bias_mIdentity_50:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_50_
Identity_51IdentityRestoreV2:tensors:51*
T0*
_output_shapes
:2
Identity_51­
AssignVariableOp_51AssignVariableOp4assignvariableop_51_adam_batch_normalization_gamma_mIdentity_51:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_51_
Identity_52IdentityRestoreV2:tensors:52*
_output_shapes
:*
T02
Identity_52Ќ
AssignVariableOp_52AssignVariableOp3assignvariableop_52_adam_batch_normalization_beta_mIdentity_52:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_52_
Identity_53IdentityRestoreV2:tensors:53*
T0*
_output_shapes
:2
Identity_53Ѓ
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_conv2d_1_kernel_mIdentity_53:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_53_
Identity_54IdentityRestoreV2:tensors:54*
T0*
_output_shapes
:2
Identity_54Ё
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_conv2d_1_bias_mIdentity_54:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_54_
Identity_55IdentityRestoreV2:tensors:55*
T0*
_output_shapes
:2
Identity_55Џ
AssignVariableOp_55AssignVariableOp6assignvariableop_55_adam_batch_normalization_1_gamma_mIdentity_55:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_55_
Identity_56IdentityRestoreV2:tensors:56*
_output_shapes
:*
T02
Identity_56Ў
AssignVariableOp_56AssignVariableOp5assignvariableop_56_adam_batch_normalization_1_beta_mIdentity_56:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_56_
Identity_57IdentityRestoreV2:tensors:57*
T0*
_output_shapes
:2
Identity_57Ѓ
AssignVariableOp_57AssignVariableOp*assignvariableop_57_adam_conv2d_3_kernel_mIdentity_57:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_57_
Identity_58IdentityRestoreV2:tensors:58*
T0*
_output_shapes
:2
Identity_58Ё
AssignVariableOp_58AssignVariableOp(assignvariableop_58_adam_conv2d_3_bias_mIdentity_58:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_58_
Identity_59IdentityRestoreV2:tensors:59*
T0*
_output_shapes
:2
Identity_59Џ
AssignVariableOp_59AssignVariableOp6assignvariableop_59_adam_batch_normalization_3_gamma_mIdentity_59:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_59_
Identity_60IdentityRestoreV2:tensors:60*
T0*
_output_shapes
:2
Identity_60Ў
AssignVariableOp_60AssignVariableOp5assignvariableop_60_adam_batch_normalization_3_beta_mIdentity_60:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_60_
Identity_61IdentityRestoreV2:tensors:61*
T0*
_output_shapes
:2
Identity_61Ѓ
AssignVariableOp_61AssignVariableOp*assignvariableop_61_adam_conv2d_4_kernel_mIdentity_61:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_61_
Identity_62IdentityRestoreV2:tensors:62*
T0*
_output_shapes
:2
Identity_62Ё
AssignVariableOp_62AssignVariableOp(assignvariableop_62_adam_conv2d_4_bias_mIdentity_62:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_62_
Identity_63IdentityRestoreV2:tensors:63*
T0*
_output_shapes
:2
Identity_63Џ
AssignVariableOp_63AssignVariableOp6assignvariableop_63_adam_batch_normalization_4_gamma_mIdentity_63:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_63_
Identity_64IdentityRestoreV2:tensors:64*
T0*
_output_shapes
:2
Identity_64Ў
AssignVariableOp_64AssignVariableOp5assignvariableop_64_adam_batch_normalization_4_beta_mIdentity_64:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_64_
Identity_65IdentityRestoreV2:tensors:65*
T0*
_output_shapes
:2
Identity_65Ѓ
AssignVariableOp_65AssignVariableOp*assignvariableop_65_adam_conv2d_2_kernel_mIdentity_65:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_65_
Identity_66IdentityRestoreV2:tensors:66*
T0*
_output_shapes
:2
Identity_66Ё
AssignVariableOp_66AssignVariableOp(assignvariableop_66_adam_conv2d_2_bias_mIdentity_66:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_66_
Identity_67IdentityRestoreV2:tensors:67*
T0*
_output_shapes
:2
Identity_67Џ
AssignVariableOp_67AssignVariableOp6assignvariableop_67_adam_batch_normalization_2_gamma_mIdentity_67:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_67_
Identity_68IdentityRestoreV2:tensors:68*
T0*
_output_shapes
:2
Identity_68Ў
AssignVariableOp_68AssignVariableOp5assignvariableop_68_adam_batch_normalization_2_beta_mIdentity_68:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_68_
Identity_69IdentityRestoreV2:tensors:69*
T0*
_output_shapes
:2
Identity_69
AssignVariableOp_69AssignVariableOp$assignvariableop_69_adam_fc_kernel_mIdentity_69:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_69_
Identity_70IdentityRestoreV2:tensors:70*
_output_shapes
:*
T02
Identity_70
AssignVariableOp_70AssignVariableOp"assignvariableop_70_adam_fc_bias_mIdentity_70:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_70_
Identity_71IdentityRestoreV2:tensors:71*
T0*
_output_shapes
:2
Identity_71 
AssignVariableOp_71AssignVariableOp'assignvariableop_71_adam_conv1_kernel_vIdentity_71:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_71_
Identity_72IdentityRestoreV2:tensors:72*
_output_shapes
:*
T02
Identity_72
AssignVariableOp_72AssignVariableOp%assignvariableop_72_adam_conv1_bias_vIdentity_72:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_72_
Identity_73IdentityRestoreV2:tensors:73*
T0*
_output_shapes
:2
Identity_73Ђ
AssignVariableOp_73AssignVariableOp)assignvariableop_73_adam_bn_conv1_gamma_vIdentity_73:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_73_
Identity_74IdentityRestoreV2:tensors:74*
_output_shapes
:*
T02
Identity_74Ё
AssignVariableOp_74AssignVariableOp(assignvariableop_74_adam_bn_conv1_beta_vIdentity_74:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_74_
Identity_75IdentityRestoreV2:tensors:75*
T0*
_output_shapes
:2
Identity_75Ё
AssignVariableOp_75AssignVariableOp(assignvariableop_75_adam_conv2d_kernel_vIdentity_75:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_75_
Identity_76IdentityRestoreV2:tensors:76*
_output_shapes
:*
T02
Identity_76
AssignVariableOp_76AssignVariableOp&assignvariableop_76_adam_conv2d_bias_vIdentity_76:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_76_
Identity_77IdentityRestoreV2:tensors:77*
_output_shapes
:*
T02
Identity_77­
AssignVariableOp_77AssignVariableOp4assignvariableop_77_adam_batch_normalization_gamma_vIdentity_77:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_77_
Identity_78IdentityRestoreV2:tensors:78*
_output_shapes
:*
T02
Identity_78Ќ
AssignVariableOp_78AssignVariableOp3assignvariableop_78_adam_batch_normalization_beta_vIdentity_78:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_78_
Identity_79IdentityRestoreV2:tensors:79*
_output_shapes
:*
T02
Identity_79Ѓ
AssignVariableOp_79AssignVariableOp*assignvariableop_79_adam_conv2d_1_kernel_vIdentity_79:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_79_
Identity_80IdentityRestoreV2:tensors:80*
_output_shapes
:*
T02
Identity_80Ё
AssignVariableOp_80AssignVariableOp(assignvariableop_80_adam_conv2d_1_bias_vIdentity_80:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_80_
Identity_81IdentityRestoreV2:tensors:81*
_output_shapes
:*
T02
Identity_81Џ
AssignVariableOp_81AssignVariableOp6assignvariableop_81_adam_batch_normalization_1_gamma_vIdentity_81:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_81_
Identity_82IdentityRestoreV2:tensors:82*
_output_shapes
:*
T02
Identity_82Ў
AssignVariableOp_82AssignVariableOp5assignvariableop_82_adam_batch_normalization_1_beta_vIdentity_82:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_82_
Identity_83IdentityRestoreV2:tensors:83*
_output_shapes
:*
T02
Identity_83Ѓ
AssignVariableOp_83AssignVariableOp*assignvariableop_83_adam_conv2d_3_kernel_vIdentity_83:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_83_
Identity_84IdentityRestoreV2:tensors:84*
T0*
_output_shapes
:2
Identity_84Ё
AssignVariableOp_84AssignVariableOp(assignvariableop_84_adam_conv2d_3_bias_vIdentity_84:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_84_
Identity_85IdentityRestoreV2:tensors:85*
_output_shapes
:*
T02
Identity_85Џ
AssignVariableOp_85AssignVariableOp6assignvariableop_85_adam_batch_normalization_3_gamma_vIdentity_85:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_85_
Identity_86IdentityRestoreV2:tensors:86*
_output_shapes
:*
T02
Identity_86Ў
AssignVariableOp_86AssignVariableOp5assignvariableop_86_adam_batch_normalization_3_beta_vIdentity_86:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_86_
Identity_87IdentityRestoreV2:tensors:87*
_output_shapes
:*
T02
Identity_87Ѓ
AssignVariableOp_87AssignVariableOp*assignvariableop_87_adam_conv2d_4_kernel_vIdentity_87:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_87_
Identity_88IdentityRestoreV2:tensors:88*
_output_shapes
:*
T02
Identity_88Ё
AssignVariableOp_88AssignVariableOp(assignvariableop_88_adam_conv2d_4_bias_vIdentity_88:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_88_
Identity_89IdentityRestoreV2:tensors:89*
T0*
_output_shapes
:2
Identity_89Џ
AssignVariableOp_89AssignVariableOp6assignvariableop_89_adam_batch_normalization_4_gamma_vIdentity_89:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_89_
Identity_90IdentityRestoreV2:tensors:90*
_output_shapes
:*
T02
Identity_90Ў
AssignVariableOp_90AssignVariableOp5assignvariableop_90_adam_batch_normalization_4_beta_vIdentity_90:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_90_
Identity_91IdentityRestoreV2:tensors:91*
T0*
_output_shapes
:2
Identity_91Ѓ
AssignVariableOp_91AssignVariableOp*assignvariableop_91_adam_conv2d_2_kernel_vIdentity_91:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_91_
Identity_92IdentityRestoreV2:tensors:92*
T0*
_output_shapes
:2
Identity_92Ё
AssignVariableOp_92AssignVariableOp(assignvariableop_92_adam_conv2d_2_bias_vIdentity_92:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_92_
Identity_93IdentityRestoreV2:tensors:93*
_output_shapes
:*
T02
Identity_93Џ
AssignVariableOp_93AssignVariableOp6assignvariableop_93_adam_batch_normalization_2_gamma_vIdentity_93:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_93_
Identity_94IdentityRestoreV2:tensors:94*
_output_shapes
:*
T02
Identity_94Ў
AssignVariableOp_94AssignVariableOp5assignvariableop_94_adam_batch_normalization_2_beta_vIdentity_94:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_94_
Identity_95IdentityRestoreV2:tensors:95*
T0*
_output_shapes
:2
Identity_95
AssignVariableOp_95AssignVariableOp$assignvariableop_95_adam_fc_kernel_vIdentity_95:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_95_
Identity_96IdentityRestoreV2:tensors:96*
_output_shapes
:*
T02
Identity_96
AssignVariableOp_96AssignVariableOp"assignvariableop_96_adam_fc_bias_vIdentity_96:output:0*
dtype0*
_output_shapes
 2
AssignVariableOp_96Ј
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:2
RestoreV2_1/tensor_names
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:2
RestoreV2_1/shape_and_slicesФ
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpД
Identity_97Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^NoOp"/device:CPU:0*
_output_shapes
: *
T02
Identity_97С
Identity_98IdentityIdentity_97:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_98"#
identity_98Identity_98:output:0*
_input_shapes
: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2
RestoreV2_1RestoreV2_12(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_27AssignVariableOp_272$
AssignVariableOpAssignVariableOp2*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_39AssignVariableOp_392*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_49AssignVariableOp_492*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_69AssignVariableOp_692*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_89AssignVariableOp_892*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_96:+ '
%
_user_specified_namefile_prefix
Љ$

P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_25966

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_25951
assignmovingavg_1_25958
identityЂ#AssignMovingAvg/AssignSubVariableOpЂAssignMovingAvg/ReadVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpЂ AssignMovingAvg_1/ReadVariableOpЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/x^
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
:@2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
:@2
ReadVariableOp_1Q
ConstConst*
valueB *
dtype0*
_output_shapes
: 2
ConstU
Const_1Const*
dtype0*
_output_shapes
: *
valueB 2	
Const_1
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*
epsilon%o:*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:2
FusedBatchNormV3W
Const_2Const*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2	
Const_2
AssignMovingAvg/sub/xConst*
valueB
 *  ?*(
_class
loc:@AssignMovingAvg/25951*
dtype0*
_output_shapes
: 2
AssignMovingAvg/sub/xЎ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/25951*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_25951*
dtype0*
_output_shapes
:@2 
AssignMovingAvg/ReadVariableOpЫ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/25951*
_output_shapes
:@2
AssignMovingAvg/sub_1Д
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/25951*
_output_shapes
:@2
AssignMovingAvg/mulџ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_25951AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/25951*
dtype0*
_output_shapes
 2%
#AssignMovingAvg/AssignSubVariableOpЃ
AssignMovingAvg_1/sub/xConst*
valueB
 *  ?**
_class 
loc:@AssignMovingAvg_1/25958*
dtype0*
_output_shapes
: 2
AssignMovingAvg_1/sub/xЖ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/25958*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_25958*
dtype0*
_output_shapes
:@2"
 AssignMovingAvg_1/ReadVariableOpз
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0**
_class 
loc:@AssignMovingAvg_1/25958*
_output_shapes
:@*
T02
AssignMovingAvg_1/sub_1О
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/25958*
_output_shapes
:@2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_25958AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/25958*
dtype0*
_output_shapes
 2'
%AssignMovingAvg_1/AssignSubVariableOpИ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::::2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_12@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp:& "
 
_user_specified_nameinputs
Є
c
G__inference_activation_5_layer_call_and_return_conditional_losses_24156

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:& "
 
_user_specified_nameinputs
ѓ
ц
C__inference_bn_conv1_layer_call_and_return_conditional_losses_25286

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: 2
LogicalAnd/x^
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
: 2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
dtype0*
_output_shapes
: 2!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
dtype0*
_output_shapes
: 2#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
T0*
U0*
is_training( *
epsilon%o:2
FusedBatchNormV3S
ConstConst*
dtype0*
_output_shapes
: *
valueB
 *Єp}?2
Constь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::::2 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_1:& "
 
_user_specified_nameinputs
А
ќ
3__inference_batch_normalization_layer_call_fn_25548

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityЂStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4**
config_proto

CPU

GPU 2J 8*
Tin	
2*/
_output_shapes
:џџџџџџџџџ *,
_gradient_op_typePartitionedCallUnused*W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_23811*
Tout
22
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ц#

C__inference_bn_conv1_layer_call_and_return_conditional_losses_25338

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_25323
assignmovingavg_1_25330
identityЂ#AssignMovingAvg/AssignSubVariableOpЂAssignMovingAvg/ReadVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpЂ AssignMovingAvg_1/ReadVariableOpЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/x^
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
: 2
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
valueB *
dtype02
ConstU
Const_1Const*
dtype0*
_output_shapes
: *
valueB 2	
Const_1
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*
epsilon%o:*K
_output_shapes9
7:џџџџџџџџџ : : : : :2
FusedBatchNormV3W
Const_2Const*
dtype0*
_output_shapes
: *
valueB
 *Єp}?2	
Const_2
AssignMovingAvg/sub/xConst*
valueB
 *  ?*(
_class
loc:@AssignMovingAvg/25323*
dtype0*
_output_shapes
: 2
AssignMovingAvg/sub/xЎ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/25323*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_25323*
dtype0*
_output_shapes
: 2 
AssignMovingAvg/ReadVariableOpЫ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
_output_shapes
: *
T0*(
_class
loc:@AssignMovingAvg/253232
AssignMovingAvg/sub_1Д
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
_output_shapes
: *
T0*(
_class
loc:@AssignMovingAvg/253232
AssignMovingAvg/mulџ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_25323AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/25323*
dtype0*
_output_shapes
 2%
#AssignMovingAvg/AssignSubVariableOpЃ
AssignMovingAvg_1/sub/xConst*
valueB
 *  ?**
_class 
loc:@AssignMovingAvg_1/25330*
dtype0*
_output_shapes
: 2
AssignMovingAvg_1/sub/xЖ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/25330*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_25330*
dtype0*
_output_shapes
: 2"
 AssignMovingAvg_1/ReadVariableOpз
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/25330*
_output_shapes
: 2
AssignMovingAvg_1/sub_1О
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0**
_class 
loc:@AssignMovingAvg_1/25330*
_output_shapes
: *
T02
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_25330AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/25330*
dtype0*
_output_shapes
 2'
%AssignMovingAvg_1/AssignSubVariableOpІ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ ::::2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
Ъ
ѓ
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_24219

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: 2
LogicalAnd/x^
LogicalAnd/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
:@2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
:@2
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
dtype0*
_output_shapes
:@2!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
dtype0*
_output_shapes
:@2#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*K
_output_shapes9
7:џџџџџџџџџ@:@:@:@:@:*
T0*
U0*
is_training( *
epsilon%o:2
FusedBatchNormV3S
ConstConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2
Constк
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ@::::2 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_1:& "
 
_user_specified_nameinputs

ѓ
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_23331

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: 2
LogicalAnd/x^
LogicalAnd/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
:@2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
:@2
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
dtype0*
_output_shapes
:@2!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
dtype0*
_output_shapes
:@2#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
is_training( *
epsilon%o:*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
T0*
U02
FusedBatchNormV3S
ConstConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2
Constь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
T02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::::2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp:& "
 
_user_specified_nameinputs
$

C__inference_bn_conv1_layer_call_and_return_conditional_losses_25264

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_25249
assignmovingavg_1_25256
identityЂ#AssignMovingAvg/AssignSubVariableOpЂAssignMovingAvg/ReadVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpЂ AssignMovingAvg_1/ReadVariableOpЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
dtype0
*
_output_shapes
: *
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
: 2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
: 2
ReadVariableOp_1Q
ConstConst*
valueB *
dtype0*
_output_shapes
: 2
ConstU
Const_1Const*
_output_shapes
: *
valueB *
dtype02	
Const_1
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
T0*
U0*
epsilon%o:2
FusedBatchNormV3W
Const_2Const*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2	
Const_2
AssignMovingAvg/sub/xConst*
valueB
 *  ?*(
_class
loc:@AssignMovingAvg/25249*
dtype0*
_output_shapes
: 2
AssignMovingAvg/sub/xЎ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
_output_shapes
: *
T0*(
_class
loc:@AssignMovingAvg/252492
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_25249*
dtype0*
_output_shapes
: 2 
AssignMovingAvg/ReadVariableOpЫ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/25249*
_output_shapes
: 2
AssignMovingAvg/sub_1Д
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/25249*
_output_shapes
: 2
AssignMovingAvg/mulџ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_25249AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *(
_class
loc:@AssignMovingAvg/25249*
dtype02%
#AssignMovingAvg/AssignSubVariableOpЃ
AssignMovingAvg_1/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?**
_class 
loc:@AssignMovingAvg_1/252562
AssignMovingAvg_1/sub/xЖ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/25256*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_25256*
dtype0*
_output_shapes
: 2"
 AssignMovingAvg_1/ReadVariableOpз
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/25256*
_output_shapes
: 2
AssignMovingAvg_1/sub_1О
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0**
_class 
loc:@AssignMovingAvg_1/25256*
_output_shapes
: *
T02
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_25256AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/25256*
dtype0*
_output_shapes
 2'
%AssignMovingAvg_1/AssignSubVariableOpИ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::::2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
Ъ
ѓ
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_23906

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: 2
LogicalAnd/x^
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
: 2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
: 2
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
dtype0*
_output_shapes
: 2#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
is_training( *
epsilon%o:*K
_output_shapes9
7:џџџџџџџџџ : : : : :*
T0*
U02
FusedBatchNormV3S
ConstConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2
Constк
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*/
_output_shapes
:џџџџџџџџџ *
T02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ ::::2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp:& "
 
_user_specified_nameinputs
а
ё
(__inference_bn_conv1_layer_call_fn_25304

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityЂStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*L
fGRE
C__inference_bn_conv1_layer_call_and_return_conditional_losses_22866*
Tout
2**
config_proto

CPU

GPU 2J 8*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
Tin	
2*,
_gradient_op_typePartitionedCallUnused2
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
T02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
с|
№
B__inference_ResNet9_layer_call_and_return_conditional_losses_24557

inputs(
$conv1_statefulpartitionedcall_args_1(
$conv1_statefulpartitionedcall_args_2+
'bn_conv1_statefulpartitionedcall_args_1+
'bn_conv1_statefulpartitionedcall_args_2+
'bn_conv1_statefulpartitionedcall_args_3+
'bn_conv1_statefulpartitionedcall_args_4)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_26
2batch_normalization_statefulpartitionedcall_args_16
2batch_normalization_statefulpartitionedcall_args_26
2batch_normalization_statefulpartitionedcall_args_36
2batch_normalization_statefulpartitionedcall_args_4+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_28
4batch_normalization_1_statefulpartitionedcall_args_18
4batch_normalization_1_statefulpartitionedcall_args_28
4batch_normalization_1_statefulpartitionedcall_args_38
4batch_normalization_1_statefulpartitionedcall_args_4+
'conv2d_3_statefulpartitionedcall_args_1+
'conv2d_3_statefulpartitionedcall_args_28
4batch_normalization_3_statefulpartitionedcall_args_18
4batch_normalization_3_statefulpartitionedcall_args_28
4batch_normalization_3_statefulpartitionedcall_args_38
4batch_normalization_3_statefulpartitionedcall_args_4+
'conv2d_4_statefulpartitionedcall_args_1+
'conv2d_4_statefulpartitionedcall_args_2+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_28
4batch_normalization_4_statefulpartitionedcall_args_18
4batch_normalization_4_statefulpartitionedcall_args_28
4batch_normalization_4_statefulpartitionedcall_args_38
4batch_normalization_4_statefulpartitionedcall_args_48
4batch_normalization_2_statefulpartitionedcall_args_18
4batch_normalization_2_statefulpartitionedcall_args_28
4batch_normalization_2_statefulpartitionedcall_args_38
4batch_normalization_2_statefulpartitionedcall_args_4%
!fc_statefulpartitionedcall_args_1%
!fc_statefulpartitionedcall_args_2
identityЂ+batch_normalization/StatefulPartitionedCallЂ-batch_normalization_1/StatefulPartitionedCallЂ-batch_normalization_2/StatefulPartitionedCallЂ-batch_normalization_3/StatefulPartitionedCallЂ-batch_normalization_4/StatefulPartitionedCallЂ bn_conv1/StatefulPartitionedCallЂconv1/StatefulPartitionedCallЂconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂ conv2d_2/StatefulPartitionedCallЂ conv2d_3/StatefulPartitionedCallЂ conv2d_4/StatefulPartitionedCallЂfc/StatefulPartitionedCallЂ
conv1/StatefulPartitionedCallStatefulPartitionedCallinputs$conv1_statefulpartitionedcall_args_1$conv1_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCallUnused*I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_22733*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ 2
conv1/StatefulPartitionedCallЅ
 bn_conv1/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0'bn_conv1_statefulpartitionedcall_args_1'bn_conv1_statefulpartitionedcall_args_2'bn_conv1_statefulpartitionedcall_args_3'bn_conv1_statefulpartitionedcall_args_4**
config_proto

CPU

GPU 2J 8*
Tin	
2*/
_output_shapes
:џџџџџџџџџ *,
_gradient_op_typePartitionedCallUnused*L
fGRE
C__inference_bn_conv1_layer_call_and_return_conditional_losses_23715*
Tout
22"
 bn_conv1/StatefulPartitionedCallю
activation/PartitionedCallPartitionedCall)bn_conv1/StatefulPartitionedCall:output:0*N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_23744*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ *,
_gradient_op_typePartitionedCallUnused2
activation/PartitionedCallё
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ *,
_gradient_op_typePartitionedCallUnused*Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_22879*
Tout
22
max_pooling2d/PartitionedCallЧ
conv2d/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*/
_output_shapes
:џџџџџџџџџ *
Tin
2*,
_gradient_op_typePartitionedCallUnused*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_22896*
Tout
2**
config_proto

CPU

GPU 2J 82 
conv2d/StatefulPartitionedCallѓ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:02batch_normalization_statefulpartitionedcall_args_12batch_normalization_statefulpartitionedcall_args_22batch_normalization_statefulpartitionedcall_args_32batch_normalization_statefulpartitionedcall_args_4*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin	
2*/
_output_shapes
:џџџџџџџџџ *,
_gradient_op_typePartitionedCallUnused*W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_238112-
+batch_normalization/StatefulPartitionedCallџ
activation_1/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCallUnused*P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_23840*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ 2
activation_1/PartitionedCallа
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCallUnused*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_23047*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ 2"
 conv2d_1/StatefulPartitionedCall
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:04batch_normalization_1_statefulpartitionedcall_args_14batch_normalization_1_statefulpartitionedcall_args_24batch_normalization_1_statefulpartitionedcall_args_34batch_normalization_1_statefulpartitionedcall_args_4*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:џџџџџџџџџ *
Tin	
2*,
_gradient_op_typePartitionedCallUnused*Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_239062/
-batch_normalization_1/StatefulPartitionedCall
activation_2/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:џџџџџџџџџ *
Tin
2*,
_gradient_op_typePartitionedCallUnused*P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_23935*
Tout
22
activation_2/PartitionedCallў
add/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0&max_pooling2d/PartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ *,
_gradient_op_typePartitionedCallUnused*G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_23949*
Tout
22
add/PartitionedCallч
activation_3/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_23963*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ *,
_gradient_op_typePartitionedCallUnused2
activation_3/PartitionedCallа
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0'conv2d_3_statefulpartitionedcall_args_1'conv2d_3_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:џџџџџџџџџ@*
Tin
2*,
_gradient_op_typePartitionedCallUnused*L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_23198*
Tout
22"
 conv2d_3/StatefulPartitionedCall
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:04batch_normalization_3_statefulpartitionedcall_args_14batch_normalization_3_statefulpartitionedcall_args_24batch_normalization_3_statefulpartitionedcall_args_34batch_normalization_3_statefulpartitionedcall_args_4*Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_24029*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:џџџџџџџџџ@*
Tin	
2*,
_gradient_op_typePartitionedCallUnused2/
-batch_normalization_3/StatefulPartitionedCall
activation_4/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*/
_output_shapes
:џџџџџџџџџ@*
Tin
2*,
_gradient_op_typePartitionedCallUnused*P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_24058*
Tout
2**
config_proto

CPU

GPU 2J 82
activation_4/PartitionedCallа
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0'conv2d_4_statefulpartitionedcall_args_1'conv2d_4_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCallUnused*L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_23349*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:џџџџџџџџџ@*
Tin
22"
 conv2d_4/StatefulPartitionedCallа
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCallUnused*L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_23500*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:џџџџџџџџџ@*
Tin
22"
 conv2d_2/StatefulPartitionedCall
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:04batch_normalization_4_statefulpartitionedcall_args_14batch_normalization_4_statefulpartitionedcall_args_24batch_normalization_4_statefulpartitionedcall_args_34batch_normalization_4_statefulpartitionedcall_args_4*Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_24127*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin	
2*/
_output_shapes
:џџџџџџџџџ@*,
_gradient_op_typePartitionedCallUnused2/
-batch_normalization_4/StatefulPartitionedCall
activation_5/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:џџџџџџџџџ@*
Tin
2*,
_gradient_op_typePartitionedCallUnused*P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_241562
activation_5/PartitionedCall
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:04batch_normalization_2_statefulpartitionedcall_args_14batch_normalization_2_statefulpartitionedcall_args_24batch_normalization_2_statefulpartitionedcall_args_34batch_normalization_2_statefulpartitionedcall_args_4*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin	
2*/
_output_shapes
:џџџџџџџџџ@*,
_gradient_op_typePartitionedCallUnused*Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_242192/
-batch_normalization_2/StatefulPartitionedCall
add_1/PartitionedCallPartitionedCall%activation_5/PartitionedCall:output:06batch_normalization_2/StatefulPartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ@*,
_gradient_op_typePartitionedCallUnused*I
fDRB
@__inference_add_1_layer_call_and_return_conditional_losses_24249*
Tout
22
add_1/PartitionedCallщ
activation_6/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ@*,
_gradient_op_typePartitionedCallUnused*P
fKRI
G__inference_activation_6_layer_call_and_return_conditional_losses_242632
activation_6/PartitionedCall
(global_average_pooling2d/PartitionedCallPartitionedCall%activation_6/PartitionedCall:output:0*
Tout
2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:џџџџџџџџџ@*
Tin
2*,
_gradient_op_typePartitionedCallUnused*\
fWRU
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_236472*
(global_average_pooling2d/PartitionedCallх
flatten/PartitionedCallPartitionedCall1global_average_pooling2d/PartitionedCall:output:0*,
_gradient_op_typePartitionedCallUnused*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_24278*
Tout
2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:џџџџџџџџџ@*
Tin
22
flatten/PartitionedCallЅ
fc/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0!fc_statefulpartitionedcall_args_1!fc_statefulpartitionedcall_args_2*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:џџџџџџџџџ*,
_gradient_op_typePartitionedCallUnused*F
fAR?
=__inference_fc_layer_call_and_return_conditional_losses_242972
fc/StatefulPartitionedCallђ
IdentityIdentity#fc/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall!^bn_conv1/StatefulPartitionedCall^conv1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^fc/StatefulPartitionedCall*'
_output_shapes
:џџџџџџџџџ*
T02

Identity"
identityIdentity:output:0*Ш
_input_shapesЖ
Г:џџџџџџџџџ		::::::::::::::::::::::::::::::::::::::28
fc/StatefulPartitionedCallfc/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 bn_conv1/StatefulPartitionedCall bn_conv1/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
Ї$

N__inference_batch_normalization_layer_call_and_return_conditional_losses_22998

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_22983
assignmovingavg_1_22990
identityЂ#AssignMovingAvg/AssignSubVariableOpЂAssignMovingAvg/ReadVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpЂ AssignMovingAvg_1/ReadVariableOpЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/x^
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Q
ConstConst*
valueB *
dtype0*
_output_shapes
: 2
ConstU
Const_1Const*
valueB *
dtype0*
_output_shapes
: 2	
Const_1
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*
epsilon%o:*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :2
FusedBatchNormV3W
Const_2Const*
dtype0*
_output_shapes
: *
valueB
 *Єp}?2	
Const_2
AssignMovingAvg/sub/xConst*
_output_shapes
: *
valueB
 *  ?*(
_class
loc:@AssignMovingAvg/22983*
dtype02
AssignMovingAvg/sub/xЎ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/22983*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_22983*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpЫ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/22983*
_output_shapes
: 2
AssignMovingAvg/sub_1Д
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/22983*
_output_shapes
: 2
AssignMovingAvg/mulџ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_22983AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/22983*
dtype0*
_output_shapes
 2%
#AssignMovingAvg/AssignSubVariableOpЃ
AssignMovingAvg_1/sub/xConst*
valueB
 *  ?**
_class 
loc:@AssignMovingAvg_1/22990*
dtype0*
_output_shapes
: 2
AssignMovingAvg_1/sub/xЖ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/22990*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_22990*
dtype0*
_output_shapes
: 2"
 AssignMovingAvg_1/ReadVariableOpз
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
_output_shapes
: *
T0**
_class 
loc:@AssignMovingAvg_1/229902
AssignMovingAvg_1/sub_1О
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/22990*
_output_shapes
: 2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_22990AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/22990*
dtype0*
_output_shapes
 2'
%AssignMovingAvg_1/AssignSubVariableOpИ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::::2$
ReadVariableOp_1ReadVariableOp_12@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:& "
 
_user_specified_nameinputs
Љ$

P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_25678

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_25663
assignmovingavg_1_25670
identityЂ#AssignMovingAvg/AssignSubVariableOpЂAssignMovingAvg/ReadVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpЂ AssignMovingAvg_1/ReadVariableOpЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
dtype0
*
_output_shapes
: *
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
: 2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Q
ConstConst*
dtype0*
_output_shapes
: *
valueB 2
ConstU
Const_1Const*
valueB *
dtype0*
_output_shapes
: 2	
Const_1
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
epsilon%o:*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
T0*
U02
FusedBatchNormV3W
Const_2Const*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2	
Const_2
AssignMovingAvg/sub/xConst*
valueB
 *  ?*(
_class
loc:@AssignMovingAvg/25663*
dtype0*
_output_shapes
: 2
AssignMovingAvg/sub/xЎ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/25663*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_25663*
dtype0*
_output_shapes
: 2 
AssignMovingAvg/ReadVariableOpЫ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
_output_shapes
: *
T0*(
_class
loc:@AssignMovingAvg/256632
AssignMovingAvg/sub_1Д
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/25663*
_output_shapes
: 2
AssignMovingAvg/mulџ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_25663AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/25663*
dtype0*
_output_shapes
 2%
#AssignMovingAvg/AssignSubVariableOpЃ
AssignMovingAvg_1/sub/xConst*
valueB
 *  ?**
_class 
loc:@AssignMovingAvg_1/25670*
dtype0*
_output_shapes
: 2
AssignMovingAvg_1/sub/xЖ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0**
_class 
loc:@AssignMovingAvg_1/25670*
_output_shapes
: *
T02
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_25670*
dtype0*
_output_shapes
: 2"
 AssignMovingAvg_1/ReadVariableOpз
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/25670*
_output_shapes
: 2
AssignMovingAvg_1/sub_1О
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/25670*
_output_shapes
: 2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_25670AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/25670*
dtype0*
_output_shapes
 2'
%AssignMovingAvg_1/AssignSubVariableOpИ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::::2$
ReadVariableOp_1ReadVariableOp_12@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:& "
 
_user_specified_nameinputs
ё#

N__inference_batch_normalization_layer_call_and_return_conditional_losses_25508

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_25493
assignmovingavg_1_25500
identityЂ#AssignMovingAvg/AssignSubVariableOpЂAssignMovingAvg/ReadVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpЂ AssignMovingAvg_1/ReadVariableOpЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
value	B
 Z*
dtype0
2
LogicalAnd/x^
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
: 2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
: 2
ReadVariableOp_1Q
ConstConst*
valueB *
dtype0*
_output_shapes
: 2
ConstU
Const_1Const*
_output_shapes
: *
valueB *
dtype02	
Const_1
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*
epsilon%o:*K
_output_shapes9
7:џџџџџџџџџ : : : : :2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
valueB
 *Єp}?*
dtype02	
Const_2
AssignMovingAvg/sub/xConst*
valueB
 *  ?*(
_class
loc:@AssignMovingAvg/25493*
dtype0*
_output_shapes
: 2
AssignMovingAvg/sub/xЎ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*(
_class
loc:@AssignMovingAvg/25493*
_output_shapes
: *
T02
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_25493*
dtype0*
_output_shapes
: 2 
AssignMovingAvg/ReadVariableOpЫ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*(
_class
loc:@AssignMovingAvg/25493*
_output_shapes
: *
T02
AssignMovingAvg/sub_1Д
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/25493*
_output_shapes
: 2
AssignMovingAvg/mulџ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_25493AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/25493*
dtype0*
_output_shapes
 2%
#AssignMovingAvg/AssignSubVariableOpЃ
AssignMovingAvg_1/sub/xConst*
valueB
 *  ?**
_class 
loc:@AssignMovingAvg_1/25500*
dtype0*
_output_shapes
: 2
AssignMovingAvg_1/sub/xЖ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/25500*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_25500*
dtype0*
_output_shapes
: 2"
 AssignMovingAvg_1/ReadVariableOpз
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/25500*
_output_shapes
: 2
AssignMovingAvg_1/sub_1О
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
_output_shapes
: *
T0**
_class 
loc:@AssignMovingAvg_1/255002
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_25500AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/25500*
dtype0*
_output_shapes
 2'
%AssignMovingAvg_1/AssignSubVariableOpІ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*/
_output_shapes
:џџџџџџџџџ *
T02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ ::::2$
ReadVariableOp_1ReadVariableOp_12@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:& "
 
_user_specified_nameinputs

ѓ
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_23633

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: 2
LogicalAnd/x^
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
:@2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
:@2
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
dtype0*
_output_shapes
:@2!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
dtype0*
_output_shapes
:@2#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
T0*
U0*
is_training( *
epsilon%o:2
FusedBatchNormV3S
ConstConst*
dtype0*
_output_shapes
: *
valueB
 *Єp}?2
Constь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::::2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp:& "
 
_user_specified_nameinputs
ѓ#

P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_25870

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_25855
assignmovingavg_1_25862
identityЂ#AssignMovingAvg/AssignSubVariableOpЂAssignMovingAvg/ReadVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpЂ AssignMovingAvg_1/ReadVariableOpЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/x^
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
:@2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
:@2
ReadVariableOp_1Q
ConstConst*
dtype0*
_output_shapes
: *
valueB 2
ConstU
Const_1Const*
valueB *
dtype0*
_output_shapes
: 2	
Const_1
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*K
_output_shapes9
7:џџџџџџџџџ@:@:@:@:@:*
T0*
U0*
epsilon%o:2
FusedBatchNormV3W
Const_2Const*
dtype0*
_output_shapes
: *
valueB
 *Єp}?2	
Const_2
AssignMovingAvg/sub/xConst*
valueB
 *  ?*(
_class
loc:@AssignMovingAvg/25855*
dtype0*
_output_shapes
: 2
AssignMovingAvg/sub/xЎ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/25855*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_25855*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpЫ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/25855*
_output_shapes
:@2
AssignMovingAvg/sub_1Д
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
_output_shapes
:@*
T0*(
_class
loc:@AssignMovingAvg/258552
AssignMovingAvg/mulџ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_25855AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/25855*
dtype0*
_output_shapes
 2%
#AssignMovingAvg/AssignSubVariableOpЃ
AssignMovingAvg_1/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?**
_class 
loc:@AssignMovingAvg_1/258622
AssignMovingAvg_1/sub/xЖ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/25862*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_25862*
dtype0*
_output_shapes
:@2"
 AssignMovingAvg_1/ReadVariableOpз
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/25862*
_output_shapes
:@2
AssignMovingAvg_1/sub_1О
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0**
_class 
loc:@AssignMovingAvg_1/25862*
_output_shapes
:@*
T02
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_25862AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/25862*
dtype0*
_output_shapes
 2'
%AssignMovingAvg_1/AssignSubVariableOpІ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*/
_output_shapes
:џџџџџџџџџ@*
T02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ@::::2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
Є
c
G__inference_activation_6_layer_call_and_return_conditional_losses_26267

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ@2
Relun
IdentityIdentityRelu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
T02

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:& "
 
_user_specified_nameinputs
џ
^
B__inference_flatten_layer_call_and_return_conditional_losses_26278

inputs
identity_
ConstConst*
valueB"џџџџ@   *
dtype0*
_output_shapes
:2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ@:& "
 
_user_specified_nameinputs
Д
ў
5__inference_batch_normalization_3_layer_call_fn_25901

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityЂStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4**
config_proto

CPU

GPU 2J 8*
Tin	
2*/
_output_shapes
:џџџџџџџџџ@*,
_gradient_op_typePartitionedCallUnused*Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_24007*
Tout
22
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Д
ў
5__inference_batch_normalization_2_layer_call_fn_26167

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityЂStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*,
_gradient_op_typePartitionedCallUnused*Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_24197*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin	
2*/
_output_shapes
:џџџџџџџџџ@2
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*/
_output_shapes
:џџџџџџџџџ@*
T02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ў
ё
N__inference_batch_normalization_layer_call_and_return_conditional_losses_23029

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: 2
LogicalAnd/x^
LogicalAnd/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
: 2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
: 2
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
dtype0*
_output_shapes
: 2!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
dtype0*
_output_shapes
: 2#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
epsilon%o:*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
T0*
U0*
is_training( 2
FusedBatchNormV3S
ConstConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2
Constь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::::2 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_1:& "
 
_user_specified_nameinputs
И
І
%__inference_conv1_layer_call_fn_22741

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tout
2**
config_proto

CPU

GPU 2J 8*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
Tin
2*,
_gradient_op_typePartitionedCallUnused*I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_227332
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Д
ў
5__inference_batch_normalization_3_layer_call_fn_25910

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityЂStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*/
_output_shapes
:џџџџџџџџџ@*,
_gradient_op_typePartitionedCallUnused*Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_24029*
Tout
2**
config_proto

CPU

GPU 2J 82
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Љ$

P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_23300

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_23285
assignmovingavg_1_23292
identityЂ#AssignMovingAvg/AssignSubVariableOpЂAssignMovingAvg/ReadVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpЂ AssignMovingAvg_1/ReadVariableOpЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
value	B
 Z*
dtype0
2
LogicalAnd/x^
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
:@2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
:@2
ReadVariableOp_1Q
ConstConst*
dtype0*
_output_shapes
: *
valueB 2
ConstU
Const_1Const*
valueB *
dtype0*
_output_shapes
: 2	
Const_1
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
epsilon%o:*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
T0*
U02
FusedBatchNormV3W
Const_2Const*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2	
Const_2
AssignMovingAvg/sub/xConst*
valueB
 *  ?*(
_class
loc:@AssignMovingAvg/23285*
dtype0*
_output_shapes
: 2
AssignMovingAvg/sub/xЎ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/23285*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_23285*
dtype0*
_output_shapes
:@2 
AssignMovingAvg/ReadVariableOpЫ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/23285*
_output_shapes
:@2
AssignMovingAvg/sub_1Д
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/23285*
_output_shapes
:@2
AssignMovingAvg/mulџ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_23285AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/23285*
dtype0*
_output_shapes
 2%
#AssignMovingAvg/AssignSubVariableOpЃ
AssignMovingAvg_1/sub/xConst*
valueB
 *  ?**
_class 
loc:@AssignMovingAvg_1/23292*
dtype0*
_output_shapes
: 2
AssignMovingAvg_1/sub/xЖ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/23292*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_23292*
dtype0*
_output_shapes
:@2"
 AssignMovingAvg_1/ReadVariableOpз
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/23292*
_output_shapes
:@2
AssignMovingAvg_1/sub_1О
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/23292*
_output_shapes
:@2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_23292AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/23292*
dtype0*
_output_shapes
 2'
%AssignMovingAvg_1/AssignSubVariableOpИ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
T02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_12@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp:& "
 
_user_specified_nameinputs
с|
№
B__inference_ResNet9_layer_call_and_return_conditional_losses_24447

inputs(
$conv1_statefulpartitionedcall_args_1(
$conv1_statefulpartitionedcall_args_2+
'bn_conv1_statefulpartitionedcall_args_1+
'bn_conv1_statefulpartitionedcall_args_2+
'bn_conv1_statefulpartitionedcall_args_3+
'bn_conv1_statefulpartitionedcall_args_4)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_26
2batch_normalization_statefulpartitionedcall_args_16
2batch_normalization_statefulpartitionedcall_args_26
2batch_normalization_statefulpartitionedcall_args_36
2batch_normalization_statefulpartitionedcall_args_4+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_28
4batch_normalization_1_statefulpartitionedcall_args_18
4batch_normalization_1_statefulpartitionedcall_args_28
4batch_normalization_1_statefulpartitionedcall_args_38
4batch_normalization_1_statefulpartitionedcall_args_4+
'conv2d_3_statefulpartitionedcall_args_1+
'conv2d_3_statefulpartitionedcall_args_28
4batch_normalization_3_statefulpartitionedcall_args_18
4batch_normalization_3_statefulpartitionedcall_args_28
4batch_normalization_3_statefulpartitionedcall_args_38
4batch_normalization_3_statefulpartitionedcall_args_4+
'conv2d_4_statefulpartitionedcall_args_1+
'conv2d_4_statefulpartitionedcall_args_2+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_28
4batch_normalization_4_statefulpartitionedcall_args_18
4batch_normalization_4_statefulpartitionedcall_args_28
4batch_normalization_4_statefulpartitionedcall_args_38
4batch_normalization_4_statefulpartitionedcall_args_48
4batch_normalization_2_statefulpartitionedcall_args_18
4batch_normalization_2_statefulpartitionedcall_args_28
4batch_normalization_2_statefulpartitionedcall_args_38
4batch_normalization_2_statefulpartitionedcall_args_4%
!fc_statefulpartitionedcall_args_1%
!fc_statefulpartitionedcall_args_2
identityЂ+batch_normalization/StatefulPartitionedCallЂ-batch_normalization_1/StatefulPartitionedCallЂ-batch_normalization_2/StatefulPartitionedCallЂ-batch_normalization_3/StatefulPartitionedCallЂ-batch_normalization_4/StatefulPartitionedCallЂ bn_conv1/StatefulPartitionedCallЂconv1/StatefulPartitionedCallЂconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂ conv2d_2/StatefulPartitionedCallЂ conv2d_3/StatefulPartitionedCallЂ conv2d_4/StatefulPartitionedCallЂfc/StatefulPartitionedCallЂ
conv1/StatefulPartitionedCallStatefulPartitionedCallinputs$conv1_statefulpartitionedcall_args_1$conv1_statefulpartitionedcall_args_2*I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_22733*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ *,
_gradient_op_typePartitionedCallUnused2
conv1/StatefulPartitionedCallЅ
 bn_conv1/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0'bn_conv1_statefulpartitionedcall_args_1'bn_conv1_statefulpartitionedcall_args_2'bn_conv1_statefulpartitionedcall_args_3'bn_conv1_statefulpartitionedcall_args_4*,
_gradient_op_typePartitionedCallUnused*L
fGRE
C__inference_bn_conv1_layer_call_and_return_conditional_losses_23693*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:џџџџџџџџџ *
Tin	
22"
 bn_conv1/StatefulPartitionedCallю
activation/PartitionedCallPartitionedCall)bn_conv1/StatefulPartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ *,
_gradient_op_typePartitionedCallUnused*N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_23744*
Tout
22
activation/PartitionedCallё
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*,
_gradient_op_typePartitionedCallUnused*Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_22879*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ 2
max_pooling2d/PartitionedCallЧ
conv2d/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCallUnused*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_22896*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:џџџџџџџџџ *
Tin
22 
conv2d/StatefulPartitionedCallѓ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:02batch_normalization_statefulpartitionedcall_args_12batch_normalization_statefulpartitionedcall_args_22batch_normalization_statefulpartitionedcall_args_32batch_normalization_statefulpartitionedcall_args_4*W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_23789*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin	
2*/
_output_shapes
:џџџџџџџџџ *,
_gradient_op_typePartitionedCallUnused2-
+batch_normalization/StatefulPartitionedCallџ
activation_1/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_23840*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:џџџџџџџџџ *
Tin
2*,
_gradient_op_typePartitionedCallUnused2
activation_1/PartitionedCallа
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ *,
_gradient_op_typePartitionedCallUnused*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_230472"
 conv2d_1/StatefulPartitionedCall
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:04batch_normalization_1_statefulpartitionedcall_args_14batch_normalization_1_statefulpartitionedcall_args_24batch_normalization_1_statefulpartitionedcall_args_34batch_normalization_1_statefulpartitionedcall_args_4**
config_proto

CPU

GPU 2J 8*
Tin	
2*/
_output_shapes
:џџџџџџџџџ *,
_gradient_op_typePartitionedCallUnused*Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_23884*
Tout
22/
-batch_normalization_1/StatefulPartitionedCall
activation_2/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_23935*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:џџџџџџџџџ *
Tin
2*,
_gradient_op_typePartitionedCallUnused2
activation_2/PartitionedCallў
add/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0&max_pooling2d/PartitionedCall:output:0*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:џџџџџџџџџ *
Tin
2*,
_gradient_op_typePartitionedCallUnused*G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_239492
add/PartitionedCallч
activation_3/PartitionedCallPartitionedCalladd/PartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:џџџџџџџџџ *
Tin
2*,
_gradient_op_typePartitionedCallUnused*P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_23963*
Tout
22
activation_3/PartitionedCallа
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0'conv2d_3_statefulpartitionedcall_args_1'conv2d_3_statefulpartitionedcall_args_2*/
_output_shapes
:џџџџџџџџџ@*
Tin
2*,
_gradient_op_typePartitionedCallUnused*L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_23198*
Tout
2**
config_proto

CPU

GPU 2J 82"
 conv2d_3/StatefulPartitionedCall
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:04batch_normalization_3_statefulpartitionedcall_args_14batch_normalization_3_statefulpartitionedcall_args_24batch_normalization_3_statefulpartitionedcall_args_34batch_normalization_3_statefulpartitionedcall_args_4*,
_gradient_op_typePartitionedCallUnused*Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_24007*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin	
2*/
_output_shapes
:џџџџџџџџџ@2/
-batch_normalization_3/StatefulPartitionedCall
activation_4/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCallUnused*P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_24058*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ@2
activation_4/PartitionedCallа
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0'conv2d_4_statefulpartitionedcall_args_1'conv2d_4_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ@*,
_gradient_op_typePartitionedCallUnused*L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_23349*
Tout
22"
 conv2d_4/StatefulPartitionedCallа
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCallUnused*L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_23500*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ@2"
 conv2d_2/StatefulPartitionedCall
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:04batch_normalization_4_statefulpartitionedcall_args_14batch_normalization_4_statefulpartitionedcall_args_24batch_normalization_4_statefulpartitionedcall_args_34batch_normalization_4_statefulpartitionedcall_args_4**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:џџџџџџџџџ@*
Tin	
2*,
_gradient_op_typePartitionedCallUnused*Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_24105*
Tout
22/
-batch_normalization_4/StatefulPartitionedCall
activation_5/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*/
_output_shapes
:џџџџџџџџџ@*,
_gradient_op_typePartitionedCallUnused*P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_24156*
Tout
2**
config_proto

CPU

GPU 2J 82
activation_5/PartitionedCall
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:04batch_normalization_2_statefulpartitionedcall_args_14batch_normalization_2_statefulpartitionedcall_args_24batch_normalization_2_statefulpartitionedcall_args_34batch_normalization_2_statefulpartitionedcall_args_4**
config_proto

CPU

GPU 2J 8*
Tin	
2*/
_output_shapes
:џџџџџџџџџ@*,
_gradient_op_typePartitionedCallUnused*Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_24197*
Tout
22/
-batch_normalization_2/StatefulPartitionedCall
add_1/PartitionedCallPartitionedCall%activation_5/PartitionedCall:output:06batch_normalization_2/StatefulPartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ@*,
_gradient_op_typePartitionedCallUnused*I
fDRB
@__inference_add_1_layer_call_and_return_conditional_losses_24249*
Tout
22
add_1/PartitionedCallщ
activation_6/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*,
_gradient_op_typePartitionedCallUnused*P
fKRI
G__inference_activation_6_layer_call_and_return_conditional_losses_24263*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:џџџџџџџџџ@*
Tin
22
activation_6/PartitionedCall
(global_average_pooling2d/PartitionedCallPartitionedCall%activation_6/PartitionedCall:output:0*,
_gradient_op_typePartitionedCallUnused*\
fWRU
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_23647*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:џџџџџџџџџ@2*
(global_average_pooling2d/PartitionedCallх
flatten/PartitionedCallPartitionedCall1global_average_pooling2d/PartitionedCall:output:0*,
_gradient_op_typePartitionedCallUnused*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_24278*
Tout
2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:џџџџџџџџџ@*
Tin
22
flatten/PartitionedCallЅ
fc/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0!fc_statefulpartitionedcall_args_1!fc_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:џџџџџџџџџ*,
_gradient_op_typePartitionedCallUnused*F
fAR?
=__inference_fc_layer_call_and_return_conditional_losses_24297*
Tout
22
fc/StatefulPartitionedCallђ
IdentityIdentity#fc/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall!^bn_conv1/StatefulPartitionedCall^conv1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^fc/StatefulPartitionedCall*'
_output_shapes
:џџџџџџџџџ*
T02

Identity"
identityIdentity:output:0*Ш
_input_shapesЖ
Г:џџџџџџџџџ		::::::::::::::::::::::::::::::::::::::28
fc/StatefulPartitionedCallfc/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 bn_conv1/StatefulPartitionedCall bn_conv1/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
ф|
ё
B__inference_ResNet9_layer_call_and_return_conditional_losses_24377
input_1(
$conv1_statefulpartitionedcall_args_1(
$conv1_statefulpartitionedcall_args_2+
'bn_conv1_statefulpartitionedcall_args_1+
'bn_conv1_statefulpartitionedcall_args_2+
'bn_conv1_statefulpartitionedcall_args_3+
'bn_conv1_statefulpartitionedcall_args_4)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_26
2batch_normalization_statefulpartitionedcall_args_16
2batch_normalization_statefulpartitionedcall_args_26
2batch_normalization_statefulpartitionedcall_args_36
2batch_normalization_statefulpartitionedcall_args_4+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_28
4batch_normalization_1_statefulpartitionedcall_args_18
4batch_normalization_1_statefulpartitionedcall_args_28
4batch_normalization_1_statefulpartitionedcall_args_38
4batch_normalization_1_statefulpartitionedcall_args_4+
'conv2d_3_statefulpartitionedcall_args_1+
'conv2d_3_statefulpartitionedcall_args_28
4batch_normalization_3_statefulpartitionedcall_args_18
4batch_normalization_3_statefulpartitionedcall_args_28
4batch_normalization_3_statefulpartitionedcall_args_38
4batch_normalization_3_statefulpartitionedcall_args_4+
'conv2d_4_statefulpartitionedcall_args_1+
'conv2d_4_statefulpartitionedcall_args_2+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_28
4batch_normalization_4_statefulpartitionedcall_args_18
4batch_normalization_4_statefulpartitionedcall_args_28
4batch_normalization_4_statefulpartitionedcall_args_38
4batch_normalization_4_statefulpartitionedcall_args_48
4batch_normalization_2_statefulpartitionedcall_args_18
4batch_normalization_2_statefulpartitionedcall_args_28
4batch_normalization_2_statefulpartitionedcall_args_38
4batch_normalization_2_statefulpartitionedcall_args_4%
!fc_statefulpartitionedcall_args_1%
!fc_statefulpartitionedcall_args_2
identityЂ+batch_normalization/StatefulPartitionedCallЂ-batch_normalization_1/StatefulPartitionedCallЂ-batch_normalization_2/StatefulPartitionedCallЂ-batch_normalization_3/StatefulPartitionedCallЂ-batch_normalization_4/StatefulPartitionedCallЂ bn_conv1/StatefulPartitionedCallЂconv1/StatefulPartitionedCallЂconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂ conv2d_2/StatefulPartitionedCallЂ conv2d_3/StatefulPartitionedCallЂ conv2d_4/StatefulPartitionedCallЂfc/StatefulPartitionedCallЃ
conv1/StatefulPartitionedCallStatefulPartitionedCallinput_1$conv1_statefulpartitionedcall_args_1$conv1_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ *,
_gradient_op_typePartitionedCallUnused*I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_22733*
Tout
22
conv1/StatefulPartitionedCallЅ
 bn_conv1/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0'bn_conv1_statefulpartitionedcall_args_1'bn_conv1_statefulpartitionedcall_args_2'bn_conv1_statefulpartitionedcall_args_3'bn_conv1_statefulpartitionedcall_args_4*,
_gradient_op_typePartitionedCallUnused*L
fGRE
C__inference_bn_conv1_layer_call_and_return_conditional_losses_23715*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:џџџџџџџџџ *
Tin	
22"
 bn_conv1/StatefulPartitionedCallю
activation/PartitionedCallPartitionedCall)bn_conv1/StatefulPartitionedCall:output:0*
Tin
2*/
_output_shapes
:џџџџџџџџџ *,
_gradient_op_typePartitionedCallUnused*N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_23744*
Tout
2**
config_proto

CPU

GPU 2J 82
activation/PartitionedCallё
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ *,
_gradient_op_typePartitionedCallUnused*Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_22879*
Tout
22
max_pooling2d/PartitionedCallЧ
conv2d/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ *,
_gradient_op_typePartitionedCallUnused*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_22896*
Tout
22 
conv2d/StatefulPartitionedCallѓ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:02batch_normalization_statefulpartitionedcall_args_12batch_normalization_statefulpartitionedcall_args_22batch_normalization_statefulpartitionedcall_args_32batch_normalization_statefulpartitionedcall_args_4*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin	
2*/
_output_shapes
:џџџџџџџџџ *,
_gradient_op_typePartitionedCallUnused*W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_238112-
+batch_normalization/StatefulPartitionedCallџ
activation_1/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCallUnused*P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_23840*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ 2
activation_1/PartitionedCallа
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCallUnused*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_23047*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ 2"
 conv2d_1/StatefulPartitionedCall
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:04batch_normalization_1_statefulpartitionedcall_args_14batch_normalization_1_statefulpartitionedcall_args_24batch_normalization_1_statefulpartitionedcall_args_34batch_normalization_1_statefulpartitionedcall_args_4*,
_gradient_op_typePartitionedCallUnused*Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_23906*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin	
2*/
_output_shapes
:џџџџџџџџџ 2/
-batch_normalization_1/StatefulPartitionedCall
activation_2/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*/
_output_shapes
:џџџџџџџџџ *
Tin
2*,
_gradient_op_typePartitionedCallUnused*P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_23935*
Tout
2**
config_proto

CPU

GPU 2J 82
activation_2/PartitionedCallў
add/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0&max_pooling2d/PartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ *,
_gradient_op_typePartitionedCallUnused*G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_23949*
Tout
22
add/PartitionedCallч
activation_3/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*,
_gradient_op_typePartitionedCallUnused*P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_23963*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ 2
activation_3/PartitionedCallа
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0'conv2d_3_statefulpartitionedcall_args_1'conv2d_3_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCallUnused*L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_23198*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ@2"
 conv2d_3/StatefulPartitionedCall
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:04batch_normalization_3_statefulpartitionedcall_args_14batch_normalization_3_statefulpartitionedcall_args_24batch_normalization_3_statefulpartitionedcall_args_34batch_normalization_3_statefulpartitionedcall_args_4*Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_24029*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:џџџџџџџџџ@*
Tin	
2*,
_gradient_op_typePartitionedCallUnused2/
-batch_normalization_3/StatefulPartitionedCall
activation_4/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*/
_output_shapes
:џџџџџџџџџ@*
Tin
2*,
_gradient_op_typePartitionedCallUnused*P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_24058*
Tout
2**
config_proto

CPU

GPU 2J 82
activation_4/PartitionedCallа
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0'conv2d_4_statefulpartitionedcall_args_1'conv2d_4_statefulpartitionedcall_args_2*/
_output_shapes
:џџџџџџџџџ@*
Tin
2*,
_gradient_op_typePartitionedCallUnused*L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_23349*
Tout
2**
config_proto

CPU

GPU 2J 82"
 conv2d_4/StatefulPartitionedCallа
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ@*,
_gradient_op_typePartitionedCallUnused*L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_235002"
 conv2d_2/StatefulPartitionedCall
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:04batch_normalization_4_statefulpartitionedcall_args_14batch_normalization_4_statefulpartitionedcall_args_24batch_normalization_4_statefulpartitionedcall_args_34batch_normalization_4_statefulpartitionedcall_args_4**
config_proto

CPU

GPU 2J 8*
Tin	
2*/
_output_shapes
:џџџџџџџџџ@*,
_gradient_op_typePartitionedCallUnused*Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_24127*
Tout
22/
-batch_normalization_4/StatefulPartitionedCall
activation_5/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ@*,
_gradient_op_typePartitionedCallUnused*P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_24156*
Tout
22
activation_5/PartitionedCall
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:04batch_normalization_2_statefulpartitionedcall_args_14batch_normalization_2_statefulpartitionedcall_args_24batch_normalization_2_statefulpartitionedcall_args_34batch_normalization_2_statefulpartitionedcall_args_4*,
_gradient_op_typePartitionedCallUnused*Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_24219*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin	
2*/
_output_shapes
:џџџџџџџџџ@2/
-batch_normalization_2/StatefulPartitionedCall
add_1/PartitionedCallPartitionedCall%activation_5/PartitionedCall:output:06batch_normalization_2/StatefulPartitionedCall:output:0*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ@*,
_gradient_op_typePartitionedCallUnused*I
fDRB
@__inference_add_1_layer_call_and_return_conditional_losses_242492
add_1/PartitionedCallщ
activation_6/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ@*,
_gradient_op_typePartitionedCallUnused*P
fKRI
G__inference_activation_6_layer_call_and_return_conditional_losses_24263*
Tout
22
activation_6/PartitionedCall
(global_average_pooling2d/PartitionedCallPartitionedCall%activation_6/PartitionedCall:output:0*'
_output_shapes
:џџџџџџџџџ@*
Tin
2*,
_gradient_op_typePartitionedCallUnused*\
fWRU
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_23647*
Tout
2**
config_proto

CPU

GPU 2J 82*
(global_average_pooling2d/PartitionedCallх
flatten/PartitionedCallPartitionedCall1global_average_pooling2d/PartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:џџџџџџџџџ@*
Tin
2*,
_gradient_op_typePartitionedCallUnused*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_24278*
Tout
22
flatten/PartitionedCallЅ
fc/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0!fc_statefulpartitionedcall_args_1!fc_statefulpartitionedcall_args_2*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:џџџџџџџџџ*,
_gradient_op_typePartitionedCallUnused*F
fAR?
=__inference_fc_layer_call_and_return_conditional_losses_242972
fc/StatefulPartitionedCallђ
IdentityIdentity#fc/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall!^bn_conv1/StatefulPartitionedCall^conv1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^fc/StatefulPartitionedCall*'
_output_shapes
:џџџџџџџџџ*
T02

Identity"
identityIdentity:output:0*Ш
_input_shapesЖ
Г:џџџџџџџџџ		::::::::::::::::::::::::::::::::::::::2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall28
fc/StatefulPartitionedCallfc/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 bn_conv1/StatefulPartitionedCall bn_conv1/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
Ч
Q
%__inference_add_1_layer_call_fn_26262
inputs_0
inputs_1
identityН
PartitionedCallPartitionedCallinputs_0inputs_1*,
_gradient_op_typePartitionedCallUnused*I
fDRB
@__inference_add_1_layer_call_and_return_conditional_losses_24249*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ@2
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*/
_output_shapes
:џџџџџџџџџ@*
T02

Identity"
identityIdentity:output:0*I
_input_shapes8
6:џџџџџџџџџ@:џџџџџџџџџ@:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
њ
ж
'__inference_ResNet9_layer_call_fn_24598
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32#
statefulpartitionedcall_args_33#
statefulpartitionedcall_args_34#
statefulpartitionedcall_args_35#
statefulpartitionedcall_args_36#
statefulpartitionedcall_args_37#
statefulpartitionedcall_args_38
identityЂStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32statefulpartitionedcall_args_33statefulpartitionedcall_args_34statefulpartitionedcall_args_35statefulpartitionedcall_args_36statefulpartitionedcall_args_37statefulpartitionedcall_args_38*,
_gradient_op_typePartitionedCallUnused*K
fFRD
B__inference_ResNet9_layer_call_and_return_conditional_losses_24557*
Tout
2**
config_proto

CPU

GPU 2J 8*2
Tin+
)2'*'
_output_shapes
:џџџџџџџџџ2
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*Ш
_input_shapesЖ
Г:џџџџџџџџџ		::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
Ш
ё
N__inference_batch_normalization_layer_call_and_return_conditional_losses_25530

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: 2
LogicalAnd/x^
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
: 2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
: 2
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
dtype0*
_output_shapes
: 2!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
dtype0*
_output_shapes
: 2#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*K
_output_shapes9
7:џџџџџџџџџ : : : : :*
T0*
U0*
is_training( *
epsilon%o:2
FusedBatchNormV3S
ConstConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2
Constк
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ ::::2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp:& "
 
_user_specified_nameinputs
Њд
Ч
B__inference_ResNet9_layer_call_and_return_conditional_losses_25132

inputs(
$conv1_conv2d_readvariableop_resource)
%conv1_biasadd_readvariableop_resource$
 bn_conv1_readvariableop_resource&
"bn_conv1_readvariableop_1_resource5
1bn_conv1_fusedbatchnormv3_readvariableop_resource7
3bn_conv1_fusedbatchnormv3_readvariableop_1_resource)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource%
!fc_matmul_readvariableop_resource&
"fc_biasadd_readvariableop_resource
identityЂ3batch_normalization/FusedBatchNormV3/ReadVariableOpЂ5batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ђ"batch_normalization/ReadVariableOpЂ$batch_normalization/ReadVariableOp_1Ђ5batch_normalization_1/FusedBatchNormV3/ReadVariableOpЂ7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ђ$batch_normalization_1/ReadVariableOpЂ&batch_normalization_1/ReadVariableOp_1Ђ5batch_normalization_2/FusedBatchNormV3/ReadVariableOpЂ7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ђ$batch_normalization_2/ReadVariableOpЂ&batch_normalization_2/ReadVariableOp_1Ђ5batch_normalization_3/FusedBatchNormV3/ReadVariableOpЂ7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ђ$batch_normalization_3/ReadVariableOpЂ&batch_normalization_3/ReadVariableOp_1Ђ5batch_normalization_4/FusedBatchNormV3/ReadVariableOpЂ7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Ђ$batch_normalization_4/ReadVariableOpЂ&batch_normalization_4/ReadVariableOp_1Ђ(bn_conv1/FusedBatchNormV3/ReadVariableOpЂ*bn_conv1/FusedBatchNormV3/ReadVariableOp_1Ђbn_conv1/ReadVariableOpЂbn_conv1/ReadVariableOp_1Ђconv1/BiasAdd/ReadVariableOpЂconv1/Conv2D/ReadVariableOpЂconv2d/BiasAdd/ReadVariableOpЂconv2d/Conv2D/ReadVariableOpЂconv2d_1/BiasAdd/ReadVariableOpЂconv2d_1/Conv2D/ReadVariableOpЂconv2d_2/BiasAdd/ReadVariableOpЂconv2d_2/Conv2D/ReadVariableOpЂconv2d_3/BiasAdd/ReadVariableOpЂconv2d_3/Conv2D/ReadVariableOpЂconv2d_4/BiasAdd/ReadVariableOpЂconv2d_4/Conv2D/ReadVariableOpЂfc/BiasAdd/ReadVariableOpЂfc/MatMul/ReadVariableOpЇ
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*
dtype0*&
_output_shapes
: 2
conv1/Conv2D/ReadVariableOpЕ
conv1/Conv2DConv2Dinputs#conv1/Conv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ *
T02
conv1/Conv2D
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
dtype0*
_output_shapes
: 2
conv1/BiasAdd/ReadVariableOp 
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv1/BiasAddp
bn_conv1/LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: 2
bn_conv1/LogicalAnd/xp
bn_conv1/LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
bn_conv1/LogicalAnd/y
bn_conv1/LogicalAnd
LogicalAndbn_conv1/LogicalAnd/x:output:0bn_conv1/LogicalAnd/y:output:0*
_output_shapes
: 2
bn_conv1/LogicalAnd
bn_conv1/ReadVariableOpReadVariableOp bn_conv1_readvariableop_resource*
dtype0*
_output_shapes
: 2
bn_conv1/ReadVariableOp
bn_conv1/ReadVariableOp_1ReadVariableOp"bn_conv1_readvariableop_1_resource*
_output_shapes
: *
dtype02
bn_conv1/ReadVariableOp_1Т
(bn_conv1/FusedBatchNormV3/ReadVariableOpReadVariableOp1bn_conv1_fusedbatchnormv3_readvariableop_resource*
dtype0*
_output_shapes
: 2*
(bn_conv1/FusedBatchNormV3/ReadVariableOpШ
*bn_conv1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3bn_conv1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02,
*bn_conv1/FusedBatchNormV3/ReadVariableOp_1
bn_conv1/FusedBatchNormV3FusedBatchNormV3conv1/BiasAdd:output:0bn_conv1/ReadVariableOp:value:0!bn_conv1/ReadVariableOp_1:value:00bn_conv1/FusedBatchNormV3/ReadVariableOp:value:02bn_conv1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*
is_training( *
epsilon%o:*K
_output_shapes9
7:џџџџџџџџџ : : : : :2
bn_conv1/FusedBatchNormV3e
bn_conv1/ConstConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2
bn_conv1/Const
activation/ReluRelubn_conv1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
activation/ReluФ
max_pooling2d/MaxPoolMaxPoolactivation/Relu:activations:0*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ 2
max_pooling2d/MaxPoolЊ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*
dtype0*&
_output_shapes
:  2
conv2d/Conv2D/ReadVariableOpа
conv2d/Conv2DConv2Dmax_pooling2d/MaxPool:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
paddingSAME*/
_output_shapes
:џџџџџџџџџ *
T0*
strides
2
conv2d/Conv2DЁ
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
dtype0*
_output_shapes
: 2
conv2d/BiasAdd/ReadVariableOpЄ
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv2d/BiasAdd
 batch_normalization/LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: 2"
 batch_normalization/LogicalAnd/x
 batch_normalization/LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2"
 batch_normalization/LogicalAnd/yМ
batch_normalization/LogicalAnd
LogicalAnd)batch_normalization/LogicalAnd/x:output:0)batch_normalization/LogicalAnd/y:output:0*
_output_shapes
: 2 
batch_normalization/LogicalAndА
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
dtype0*
_output_shapes
: 2$
"batch_normalization/ReadVariableOpЖ
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
dtype0*
_output_shapes
: 2&
$batch_normalization/ReadVariableOp_1у
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
dtype0*
_output_shapes
: 25
3batch_normalization/FusedBatchNormV3/ReadVariableOpщ
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
dtype0*
_output_shapes
: 27
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1г
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*
is_training( *
epsilon%o:*K
_output_shapes9
7:џџџџџџџџџ : : : : :2&
$batch_normalization/FusedBatchNormV3{
batch_normalization/ConstConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2
batch_normalization/Const
activation_1/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*/
_output_shapes
:џџџџџџџџџ *
T02
activation_1/ReluА
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*
dtype0*&
_output_shapes
:  2 
conv2d_1/Conv2D/ReadVariableOpз
conv2d_1/Conv2DConv2Dactivation_1/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*/
_output_shapes
:џџџџџџџџџ *
T0*
strides
*
paddingSAME2
conv2d_1/Conv2DЇ
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
dtype0*
_output_shapes
: 2!
conv2d_1/BiasAdd/ReadVariableOpЌ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:џџџџџџџџџ *
T02
conv2d_1/BiasAdd
"batch_normalization_1/LogicalAnd/xConst*
_output_shapes
: *
value	B
 Z *
dtype0
2$
"batch_normalization_1/LogicalAnd/x
"batch_normalization_1/LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2$
"batch_normalization_1/LogicalAnd/yФ
 batch_normalization_1/LogicalAnd
LogicalAnd+batch_normalization_1/LogicalAnd/x:output:0+batch_normalization_1/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_1/LogicalAndЖ
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
dtype0*
_output_shapes
: 2&
$batch_normalization_1/ReadVariableOpМ
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
dtype0*
_output_shapes
: 2(
&batch_normalization_1/ReadVariableOp_1щ
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
dtype0*
_output_shapes
: 27
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
dtype0*
_output_shapes
: 29
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1с
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
epsilon%o:*K
_output_shapes9
7:џџџџџџџџџ : : : : :*
T0*
U0*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3
batch_normalization_1/ConstConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2
batch_normalization_1/Const
activation_2/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*/
_output_shapes
:џџџџџџџџџ *
T02
activation_2/Relu
add/addAddV2activation_2/Relu:activations:0max_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2	
add/addu
activation_3/ReluReluadd/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
activation_3/ReluА
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_3/Conv2D/ReadVariableOpз
conv2d_3/Conv2DConv2Dactivation_3/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ@*
T02
conv2d_3/Conv2DЇ
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
dtype0*
_output_shapes
:@2!
conv2d_3/BiasAdd/ReadVariableOpЌ
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv2d_3/BiasAdd
"batch_normalization_3/LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: 2$
"batch_normalization_3/LogicalAnd/x
"batch_normalization_3/LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2$
"batch_normalization_3/LogicalAnd/yФ
 batch_normalization_3/LogicalAnd
LogicalAnd+batch_normalization_3/LogicalAnd/x:output:0+batch_normalization_3/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_3/LogicalAndЖ
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
dtype0*
_output_shapes
:@2&
$batch_normalization_3/ReadVariableOpМ
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
dtype0*
_output_shapes
:@2(
&batch_normalization_3/ReadVariableOp_1щ
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
dtype0*
_output_shapes
:@27
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
dtype0*
_output_shapes
:@29
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1с
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_3/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*K
_output_shapes9
7:џџџџџџџџџ@:@:@:@:@:*
T0*
U0*
is_training( *
epsilon%o:2(
&batch_normalization_3/FusedBatchNormV3
batch_normalization_3/ConstConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2
batch_normalization_3/Const
activation_4/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
activation_4/ReluА
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*
dtype0*&
_output_shapes
:@@2 
conv2d_4/Conv2D/ReadVariableOpз
conv2d_4/Conv2DConv2Dactivation_4/Relu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ@*
T02
conv2d_4/Conv2DЇ
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
dtype0*
_output_shapes
:@2!
conv2d_4/BiasAdd/ReadVariableOpЌ
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv2d_4/BiasAddА
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*
dtype0*&
_output_shapes
: @2 
conv2d_2/Conv2D/ReadVariableOpз
conv2d_2/Conv2DConv2Dactivation_3/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
paddingSAME*/
_output_shapes
:џџџџџџџџџ@*
T0*
strides
2
conv2d_2/Conv2DЇ
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
dtype0*
_output_shapes
:@2!
conv2d_2/BiasAdd/ReadVariableOpЌ
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:џџџџџџџџџ@*
T02
conv2d_2/BiasAdd
"batch_normalization_4/LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: 2$
"batch_normalization_4/LogicalAnd/x
"batch_normalization_4/LogicalAnd/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z2$
"batch_normalization_4/LogicalAnd/yФ
 batch_normalization_4/LogicalAnd
LogicalAnd+batch_normalization_4/LogicalAnd/x:output:0+batch_normalization_4/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_4/LogicalAndЖ
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
dtype0*
_output_shapes
:@2&
$batch_normalization_4/ReadVariableOpМ
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
dtype0*
_output_shapes
:@2(
&batch_normalization_4/ReadVariableOp_1щ
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
dtype0*
_output_shapes
:@27
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
dtype0*
_output_shapes
:@29
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1с
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_4/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*K
_output_shapes9
7:џџџџџџџџџ@:@:@:@:@:*
T0*
U0*
is_training( *
epsilon%o:2(
&batch_normalization_4/FusedBatchNormV3
batch_normalization_4/ConstConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2
batch_normalization_4/Const
activation_5/ReluRelu*batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
activation_5/Relu
"batch_normalization_2/LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: 2$
"batch_normalization_2/LogicalAnd/x
"batch_normalization_2/LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2$
"batch_normalization_2/LogicalAnd/yФ
 batch_normalization_2/LogicalAnd
LogicalAnd+batch_normalization_2/LogicalAnd/x:output:0+batch_normalization_2/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_2/LogicalAndЖ
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
dtype0*
_output_shapes
:@2&
$batch_normalization_2/ReadVariableOpМ
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
dtype0*
_output_shapes
:@2(
&batch_normalization_2/ReadVariableOp_1щ
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
dtype0*
_output_shapes
:@27
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
dtype0*
_output_shapes
:@29
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1с
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*
is_training( *
epsilon%o:*K
_output_shapes9
7:џџџџџџџџџ@:@:@:@:@:2(
&batch_normalization_2/FusedBatchNormV3
batch_normalization_2/ConstConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2
batch_normalization_2/ConstІ
	add_1/addAddV2activation_5/Relu:activations:0*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
	add_1/addw
activation_6/ReluReluadd_1/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
activation_6/ReluГ
/global_average_pooling2d/Mean/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB"      21
/global_average_pooling2d/Mean/reduction_indicesг
global_average_pooling2d/MeanMeanactivation_6/Relu:activations:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
global_average_pooling2d/Meano
flatten/ConstConst*
_output_shapes
:*
valueB"џџџџ@   *
dtype02
flatten/Const
flatten/ReshapeReshape&global_average_pooling2d/Mean:output:0flatten/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
flatten/Reshape
fc/MatMul/ReadVariableOpReadVariableOp!fc_matmul_readvariableop_resource*
dtype0*
_output_shapes

:@2
fc/MatMul/ReadVariableOp
	fc/MatMulMatMulflatten/Reshape:output:0 fc/MatMul/ReadVariableOp:value:0*'
_output_shapes
:џџџџџџџџџ*
T02
	fc/MatMul
fc/BiasAdd/ReadVariableOpReadVariableOp"fc_biasadd_readvariableop_resource*
dtype0*
_output_shapes
:2
fc/BiasAdd/ReadVariableOp

fc/BiasAddBiasAddfc/MatMul:product:0!fc/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2

fc/BiasAddj

fc/SigmoidSigmoidfc/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

fc/Sigmoidё
IdentityIdentityfc/Sigmoid:y:04^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1)^bn_conv1/FusedBatchNormV3/ReadVariableOp+^bn_conv1/FusedBatchNormV3/ReadVariableOp_1^bn_conv1/ReadVariableOp^bn_conv1/ReadVariableOp_1^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^fc/BiasAdd/ReadVariableOp^fc/MatMul/ReadVariableOp*'
_output_shapes
:џџџџџџџџџ*
T02

Identity"
identityIdentity:output:0*Ш
_input_shapesЖ
Г:џџџџџџџџџ		::::::::::::::::::::::::::::::::::::::2L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2X
*bn_conv1/FusedBatchNormV3/ReadVariableOp_1*bn_conv1/FusedBatchNormV3/ReadVariableOp_126
bn_conv1/ReadVariableOp_1bn_conv1/ReadVariableOp_12B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp22
bn_conv1/ReadVariableOpbn_conv1/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2T
(bn_conv1/FusedBatchNormV3/ReadVariableOp(bn_conv1/FusedBatchNormV3/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp24
fc/MatMul/ReadVariableOpfc/MatMul/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp26
fc/BiasAdd/ReadVariableOpfc/BiasAdd/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:& "
 
_user_specified_nameinputs
ц
ќ
3__inference_batch_normalization_layer_call_fn_25465

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityЂStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4**
config_proto

CPU

GPU 2J 8*
Tin	
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *,
_gradient_op_typePartitionedCallUnused*W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_22998*
Tout
22
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Ђ
a
E__inference_activation_layer_call_and_return_conditional_losses_23744

inputs
identityV
ReluReluinputs*/
_output_shapes
:џџџџџџџџџ *
T02
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ :& "
 
_user_specified_nameinputs


м
C__inference_conv2d_2_layer_call_and_return_conditional_losses_23500

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*
dtype0*&
_output_shapes
: @2
Conv2D/ReadVariableOpЕ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
T0*
strides
*
paddingSAME2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
dtype0*
_output_shapes
:@2
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2	
BiasAddЏ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs
ъ
ў
5__inference_batch_normalization_3_layer_call_fn_25836

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityЂStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4**
config_proto

CPU

GPU 2J 8*
Tin	
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*,
_gradient_op_typePartitionedCallUnused*Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_23331*
Tout
22
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Є
c
G__inference_activation_4_layer_call_and_return_conditional_losses_25915

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:& "
 
_user_specified_nameinputs
ц
ќ
3__inference_batch_normalization_layer_call_fn_25474

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityЂStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_23029*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin	
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *,
_gradient_op_typePartitionedCallUnused2
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ё
H
,__inference_activation_5_layer_call_fn_26090

inputs
identityЗ
PartitionedCallPartitionedCallinputs*,
_gradient_op_typePartitionedCallUnused*P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_24156*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ@2
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:& "
 
_user_specified_nameinputs
Д
ў
5__inference_batch_normalization_4_layer_call_fn_26080

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityЂStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4**
config_proto

CPU

GPU 2J 8*
Tin	
2*/
_output_shapes
:џџџџџџџџџ@*,
_gradient_op_typePartitionedCallUnused*Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_24127*
Tout
22
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ѓ#

P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_26040

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_26025
assignmovingavg_1_26032
identityЂ#AssignMovingAvg/AssignSubVariableOpЂAssignMovingAvg/ReadVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpЂ AssignMovingAvg_1/ReadVariableOpЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/x^
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
:@2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
:@2
ReadVariableOp_1Q
ConstConst*
valueB *
dtype0*
_output_shapes
: 2
ConstU
Const_1Const*
valueB *
dtype0*
_output_shapes
: 2	
Const_1
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*K
_output_shapes9
7:џџџџџџџџџ@:@:@:@:@:*
T0*
U0*
epsilon%o:2
FusedBatchNormV3W
Const_2Const*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2	
Const_2
AssignMovingAvg/sub/xConst*
valueB
 *  ?*(
_class
loc:@AssignMovingAvg/26025*
dtype0*
_output_shapes
: 2
AssignMovingAvg/sub/xЎ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/26025*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_26025*
dtype0*
_output_shapes
:@2 
AssignMovingAvg/ReadVariableOpЫ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/26025*
_output_shapes
:@2
AssignMovingAvg/sub_1Д
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/26025*
_output_shapes
:@2
AssignMovingAvg/mulџ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_26025AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/26025*
dtype0*
_output_shapes
 2%
#AssignMovingAvg/AssignSubVariableOpЃ
AssignMovingAvg_1/sub/xConst*
valueB
 *  ?**
_class 
loc:@AssignMovingAvg_1/26032*
dtype0*
_output_shapes
: 2
AssignMovingAvg_1/sub/xЖ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/26032*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_26032*
dtype0*
_output_shapes
:@2"
 AssignMovingAvg_1/ReadVariableOpз
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/26032*
_output_shapes
:@2
AssignMovingAvg_1/sub_1О
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/26032*
_output_shapes
:@2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_26032AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/26032*
dtype0*
_output_shapes
 2'
%AssignMovingAvg_1/AssignSubVariableOpІ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ@::::2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_12@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp:& "
 
_user_specified_nameinputs
а
ё
(__inference_bn_conv1_layer_call_fn_25295

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityЂStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin	
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *,
_gradient_op_typePartitionedCallUnused*L
fGRE
C__inference_bn_conv1_layer_call_and_return_conditional_losses_228352
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ф
Ѓ
"__inference_fc_layer_call_fn_26301

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCallUnused*F
fAR?
=__inference_fc_layer_call_and_return_conditional_losses_24297*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:џџџџџџџџџ2
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ф|
ё
B__inference_ResNet9_layer_call_and_return_conditional_losses_24310
input_1(
$conv1_statefulpartitionedcall_args_1(
$conv1_statefulpartitionedcall_args_2+
'bn_conv1_statefulpartitionedcall_args_1+
'bn_conv1_statefulpartitionedcall_args_2+
'bn_conv1_statefulpartitionedcall_args_3+
'bn_conv1_statefulpartitionedcall_args_4)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_26
2batch_normalization_statefulpartitionedcall_args_16
2batch_normalization_statefulpartitionedcall_args_26
2batch_normalization_statefulpartitionedcall_args_36
2batch_normalization_statefulpartitionedcall_args_4+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_28
4batch_normalization_1_statefulpartitionedcall_args_18
4batch_normalization_1_statefulpartitionedcall_args_28
4batch_normalization_1_statefulpartitionedcall_args_38
4batch_normalization_1_statefulpartitionedcall_args_4+
'conv2d_3_statefulpartitionedcall_args_1+
'conv2d_3_statefulpartitionedcall_args_28
4batch_normalization_3_statefulpartitionedcall_args_18
4batch_normalization_3_statefulpartitionedcall_args_28
4batch_normalization_3_statefulpartitionedcall_args_38
4batch_normalization_3_statefulpartitionedcall_args_4+
'conv2d_4_statefulpartitionedcall_args_1+
'conv2d_4_statefulpartitionedcall_args_2+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_28
4batch_normalization_4_statefulpartitionedcall_args_18
4batch_normalization_4_statefulpartitionedcall_args_28
4batch_normalization_4_statefulpartitionedcall_args_38
4batch_normalization_4_statefulpartitionedcall_args_48
4batch_normalization_2_statefulpartitionedcall_args_18
4batch_normalization_2_statefulpartitionedcall_args_28
4batch_normalization_2_statefulpartitionedcall_args_38
4batch_normalization_2_statefulpartitionedcall_args_4%
!fc_statefulpartitionedcall_args_1%
!fc_statefulpartitionedcall_args_2
identityЂ+batch_normalization/StatefulPartitionedCallЂ-batch_normalization_1/StatefulPartitionedCallЂ-batch_normalization_2/StatefulPartitionedCallЂ-batch_normalization_3/StatefulPartitionedCallЂ-batch_normalization_4/StatefulPartitionedCallЂ bn_conv1/StatefulPartitionedCallЂconv1/StatefulPartitionedCallЂconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂ conv2d_2/StatefulPartitionedCallЂ conv2d_3/StatefulPartitionedCallЂ conv2d_4/StatefulPartitionedCallЂfc/StatefulPartitionedCallЃ
conv1/StatefulPartitionedCallStatefulPartitionedCallinput_1$conv1_statefulpartitionedcall_args_1$conv1_statefulpartitionedcall_args_2*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ *,
_gradient_op_typePartitionedCallUnused*I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_227332
conv1/StatefulPartitionedCallЅ
 bn_conv1/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0'bn_conv1_statefulpartitionedcall_args_1'bn_conv1_statefulpartitionedcall_args_2'bn_conv1_statefulpartitionedcall_args_3'bn_conv1_statefulpartitionedcall_args_4*,
_gradient_op_typePartitionedCallUnused*L
fGRE
C__inference_bn_conv1_layer_call_and_return_conditional_losses_23693*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin	
2*/
_output_shapes
:џџџџџџџџџ 2"
 bn_conv1/StatefulPartitionedCallю
activation/PartitionedCallPartitionedCall)bn_conv1/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCallUnused*N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_23744*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ 2
activation/PartitionedCallё
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*,
_gradient_op_typePartitionedCallUnused*Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_22879*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ 2
max_pooling2d/PartitionedCallЧ
conv2d/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCallUnused*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_22896*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:џџџџџџџџџ *
Tin
22 
conv2d/StatefulPartitionedCallѓ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:02batch_normalization_statefulpartitionedcall_args_12batch_normalization_statefulpartitionedcall_args_22batch_normalization_statefulpartitionedcall_args_32batch_normalization_statefulpartitionedcall_args_4*,
_gradient_op_typePartitionedCallUnused*W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_23789*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin	
2*/
_output_shapes
:џџџџџџџџџ 2-
+batch_normalization/StatefulPartitionedCallџ
activation_1/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ *,
_gradient_op_typePartitionedCallUnused*P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_23840*
Tout
22
activation_1/PartitionedCallа
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCallUnused*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_23047*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ 2"
 conv2d_1/StatefulPartitionedCall
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:04batch_normalization_1_statefulpartitionedcall_args_14batch_normalization_1_statefulpartitionedcall_args_24batch_normalization_1_statefulpartitionedcall_args_34batch_normalization_1_statefulpartitionedcall_args_4**
config_proto

CPU

GPU 2J 8*
Tin	
2*/
_output_shapes
:џџџџџџџџџ *,
_gradient_op_typePartitionedCallUnused*Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_23884*
Tout
22/
-batch_normalization_1/StatefulPartitionedCall
activation_2/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ *,
_gradient_op_typePartitionedCallUnused*P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_23935*
Tout
22
activation_2/PartitionedCallў
add/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0&max_pooling2d/PartitionedCall:output:0*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ *,
_gradient_op_typePartitionedCallUnused*G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_239492
add/PartitionedCallч
activation_3/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ *,
_gradient_op_typePartitionedCallUnused*P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_239632
activation_3/PartitionedCallа
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0'conv2d_3_statefulpartitionedcall_args_1'conv2d_3_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCallUnused*L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_23198*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:џџџџџџџџџ@*
Tin
22"
 conv2d_3/StatefulPartitionedCall
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:04batch_normalization_3_statefulpartitionedcall_args_14batch_normalization_3_statefulpartitionedcall_args_24batch_normalization_3_statefulpartitionedcall_args_34batch_normalization_3_statefulpartitionedcall_args_4*,
_gradient_op_typePartitionedCallUnused*Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_24007*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:џџџџџџџџџ@*
Tin	
22/
-batch_normalization_3/StatefulPartitionedCall
activation_4/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCallUnused*P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_24058*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ@2
activation_4/PartitionedCallа
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0'conv2d_4_statefulpartitionedcall_args_1'conv2d_4_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCallUnused*L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_23349*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:џџџџџџџџџ@*
Tin
22"
 conv2d_4/StatefulPartitionedCallа
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCallUnused*L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_23500*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ@2"
 conv2d_2/StatefulPartitionedCall
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:04batch_normalization_4_statefulpartitionedcall_args_14batch_normalization_4_statefulpartitionedcall_args_24batch_normalization_4_statefulpartitionedcall_args_34batch_normalization_4_statefulpartitionedcall_args_4*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:џџџџџџџџџ@*
Tin	
2*,
_gradient_op_typePartitionedCallUnused*Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_241052/
-batch_normalization_4/StatefulPartitionedCall
activation_5/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCallUnused*P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_24156*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ@2
activation_5/PartitionedCall
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:04batch_normalization_2_statefulpartitionedcall_args_14batch_normalization_2_statefulpartitionedcall_args_24batch_normalization_2_statefulpartitionedcall_args_34batch_normalization_2_statefulpartitionedcall_args_4*,
_gradient_op_typePartitionedCallUnused*Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_24197*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin	
2*/
_output_shapes
:џџџџџџџџџ@2/
-batch_normalization_2/StatefulPartitionedCall
add_1/PartitionedCallPartitionedCall%activation_5/PartitionedCall:output:06batch_normalization_2/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCallUnused*I
fDRB
@__inference_add_1_layer_call_and_return_conditional_losses_24249*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ@2
add_1/PartitionedCallщ
activation_6/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*,
_gradient_op_typePartitionedCallUnused*P
fKRI
G__inference_activation_6_layer_call_and_return_conditional_losses_24263*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ@2
activation_6/PartitionedCall
(global_average_pooling2d/PartitionedCallPartitionedCall%activation_6/PartitionedCall:output:0*
Tout
2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:џџџџџџџџџ@*
Tin
2*,
_gradient_op_typePartitionedCallUnused*\
fWRU
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_236472*
(global_average_pooling2d/PartitionedCallх
flatten/PartitionedCallPartitionedCall1global_average_pooling2d/PartitionedCall:output:0*,
_gradient_op_typePartitionedCallUnused*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_24278*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:џџџџџџџџџ@2
flatten/PartitionedCallЅ
fc/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0!fc_statefulpartitionedcall_args_1!fc_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:џџџџџџџџџ*
Tin
2*,
_gradient_op_typePartitionedCallUnused*F
fAR?
=__inference_fc_layer_call_and_return_conditional_losses_24297*
Tout
22
fc/StatefulPartitionedCallђ
IdentityIdentity#fc/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall!^bn_conv1/StatefulPartitionedCall^conv1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^fc/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*Ш
_input_shapesЖ
Г:џџџџџџџџџ		::::::::::::::::::::::::::::::::::::::2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall28
fc/StatefulPartitionedCallfc/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 bn_conv1/StatefulPartitionedCall bn_conv1/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
Є
c
G__inference_activation_3_layer_call_and_return_conditional_losses_25745

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ :& "
 
_user_specified_nameinputs
є
!
 __inference__wrapped_model_22722
input_10
,resnet9_conv1_conv2d_readvariableop_resource1
-resnet9_conv1_biasadd_readvariableop_resource,
(resnet9_bn_conv1_readvariableop_resource.
*resnet9_bn_conv1_readvariableop_1_resource=
9resnet9_bn_conv1_fusedbatchnormv3_readvariableop_resource?
;resnet9_bn_conv1_fusedbatchnormv3_readvariableop_1_resource1
-resnet9_conv2d_conv2d_readvariableop_resource2
.resnet9_conv2d_biasadd_readvariableop_resource7
3resnet9_batch_normalization_readvariableop_resource9
5resnet9_batch_normalization_readvariableop_1_resourceH
Dresnet9_batch_normalization_fusedbatchnormv3_readvariableop_resourceJ
Fresnet9_batch_normalization_fusedbatchnormv3_readvariableop_1_resource3
/resnet9_conv2d_1_conv2d_readvariableop_resource4
0resnet9_conv2d_1_biasadd_readvariableop_resource9
5resnet9_batch_normalization_1_readvariableop_resource;
7resnet9_batch_normalization_1_readvariableop_1_resourceJ
Fresnet9_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceL
Hresnet9_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource3
/resnet9_conv2d_3_conv2d_readvariableop_resource4
0resnet9_conv2d_3_biasadd_readvariableop_resource9
5resnet9_batch_normalization_3_readvariableop_resource;
7resnet9_batch_normalization_3_readvariableop_1_resourceJ
Fresnet9_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceL
Hresnet9_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource3
/resnet9_conv2d_4_conv2d_readvariableop_resource4
0resnet9_conv2d_4_biasadd_readvariableop_resource3
/resnet9_conv2d_2_conv2d_readvariableop_resource4
0resnet9_conv2d_2_biasadd_readvariableop_resource9
5resnet9_batch_normalization_4_readvariableop_resource;
7resnet9_batch_normalization_4_readvariableop_1_resourceJ
Fresnet9_batch_normalization_4_fusedbatchnormv3_readvariableop_resourceL
Hresnet9_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource9
5resnet9_batch_normalization_2_readvariableop_resource;
7resnet9_batch_normalization_2_readvariableop_1_resourceJ
Fresnet9_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceL
Hresnet9_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource-
)resnet9_fc_matmul_readvariableop_resource.
*resnet9_fc_biasadd_readvariableop_resource
identityЂ;ResNet9/batch_normalization/FusedBatchNormV3/ReadVariableOpЂ=ResNet9/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ђ*ResNet9/batch_normalization/ReadVariableOpЂ,ResNet9/batch_normalization/ReadVariableOp_1Ђ=ResNet9/batch_normalization_1/FusedBatchNormV3/ReadVariableOpЂ?ResNet9/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ђ,ResNet9/batch_normalization_1/ReadVariableOpЂ.ResNet9/batch_normalization_1/ReadVariableOp_1Ђ=ResNet9/batch_normalization_2/FusedBatchNormV3/ReadVariableOpЂ?ResNet9/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ђ,ResNet9/batch_normalization_2/ReadVariableOpЂ.ResNet9/batch_normalization_2/ReadVariableOp_1Ђ=ResNet9/batch_normalization_3/FusedBatchNormV3/ReadVariableOpЂ?ResNet9/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ђ,ResNet9/batch_normalization_3/ReadVariableOpЂ.ResNet9/batch_normalization_3/ReadVariableOp_1Ђ=ResNet9/batch_normalization_4/FusedBatchNormV3/ReadVariableOpЂ?ResNet9/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Ђ,ResNet9/batch_normalization_4/ReadVariableOpЂ.ResNet9/batch_normalization_4/ReadVariableOp_1Ђ0ResNet9/bn_conv1/FusedBatchNormV3/ReadVariableOpЂ2ResNet9/bn_conv1/FusedBatchNormV3/ReadVariableOp_1ЂResNet9/bn_conv1/ReadVariableOpЂ!ResNet9/bn_conv1/ReadVariableOp_1Ђ$ResNet9/conv1/BiasAdd/ReadVariableOpЂ#ResNet9/conv1/Conv2D/ReadVariableOpЂ%ResNet9/conv2d/BiasAdd/ReadVariableOpЂ$ResNet9/conv2d/Conv2D/ReadVariableOpЂ'ResNet9/conv2d_1/BiasAdd/ReadVariableOpЂ&ResNet9/conv2d_1/Conv2D/ReadVariableOpЂ'ResNet9/conv2d_2/BiasAdd/ReadVariableOpЂ&ResNet9/conv2d_2/Conv2D/ReadVariableOpЂ'ResNet9/conv2d_3/BiasAdd/ReadVariableOpЂ&ResNet9/conv2d_3/Conv2D/ReadVariableOpЂ'ResNet9/conv2d_4/BiasAdd/ReadVariableOpЂ&ResNet9/conv2d_4/Conv2D/ReadVariableOpЂ!ResNet9/fc/BiasAdd/ReadVariableOpЂ ResNet9/fc/MatMul/ReadVariableOpП
#ResNet9/conv1/Conv2D/ReadVariableOpReadVariableOp,resnet9_conv1_conv2d_readvariableop_resource*
dtype0*&
_output_shapes
: 2%
#ResNet9/conv1/Conv2D/ReadVariableOpЮ
ResNet9/conv1/Conv2DConv2Dinput_1+ResNet9/conv1/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ 2
ResNet9/conv1/Conv2DЖ
$ResNet9/conv1/BiasAdd/ReadVariableOpReadVariableOp-resnet9_conv1_biasadd_readvariableop_resource*
dtype0*
_output_shapes
: 2&
$ResNet9/conv1/BiasAdd/ReadVariableOpР
ResNet9/conv1/BiasAddBiasAddResNet9/conv1/Conv2D:output:0,ResNet9/conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
ResNet9/conv1/BiasAdd
ResNet9/bn_conv1/LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: 2
ResNet9/bn_conv1/LogicalAnd/x
ResNet9/bn_conv1/LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
ResNet9/bn_conv1/LogicalAnd/yА
ResNet9/bn_conv1/LogicalAnd
LogicalAnd&ResNet9/bn_conv1/LogicalAnd/x:output:0&ResNet9/bn_conv1/LogicalAnd/y:output:0*
_output_shapes
: 2
ResNet9/bn_conv1/LogicalAndЇ
ResNet9/bn_conv1/ReadVariableOpReadVariableOp(resnet9_bn_conv1_readvariableop_resource*
dtype0*
_output_shapes
: 2!
ResNet9/bn_conv1/ReadVariableOp­
!ResNet9/bn_conv1/ReadVariableOp_1ReadVariableOp*resnet9_bn_conv1_readvariableop_1_resource*
dtype0*
_output_shapes
: 2#
!ResNet9/bn_conv1/ReadVariableOp_1к
0ResNet9/bn_conv1/FusedBatchNormV3/ReadVariableOpReadVariableOp9resnet9_bn_conv1_fusedbatchnormv3_readvariableop_resource*
dtype0*
_output_shapes
: 22
0ResNet9/bn_conv1/FusedBatchNormV3/ReadVariableOpр
2ResNet9/bn_conv1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;resnet9_bn_conv1_fusedbatchnormv3_readvariableop_1_resource*
dtype0*
_output_shapes
: 24
2ResNet9/bn_conv1/FusedBatchNormV3/ReadVariableOp_1Ш
!ResNet9/bn_conv1/FusedBatchNormV3FusedBatchNormV3ResNet9/conv1/BiasAdd:output:0'ResNet9/bn_conv1/ReadVariableOp:value:0)ResNet9/bn_conv1/ReadVariableOp_1:value:08ResNet9/bn_conv1/FusedBatchNormV3/ReadVariableOp:value:0:ResNet9/bn_conv1/FusedBatchNormV3/ReadVariableOp_1:value:0*K
_output_shapes9
7:џџџџџџџџџ : : : : :*
T0*
U0*
is_training( *
epsilon%o:2#
!ResNet9/bn_conv1/FusedBatchNormV3u
ResNet9/bn_conv1/ConstConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2
ResNet9/bn_conv1/Const
ResNet9/activation/ReluRelu%ResNet9/bn_conv1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
ResNet9/activation/Reluм
ResNet9/max_pooling2d/MaxPoolMaxPool%ResNet9/activation/Relu:activations:0*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ 2
ResNet9/max_pooling2d/MaxPoolТ
$ResNet9/conv2d/Conv2D/ReadVariableOpReadVariableOp-resnet9_conv2d_conv2d_readvariableop_resource*
dtype0*&
_output_shapes
:  2&
$ResNet9/conv2d/Conv2D/ReadVariableOp№
ResNet9/conv2d/Conv2DConv2D&ResNet9/max_pooling2d/MaxPool:output:0,ResNet9/conv2d/Conv2D/ReadVariableOp:value:0*
paddingSAME*/
_output_shapes
:џџџџџџџџџ *
T0*
strides
2
ResNet9/conv2d/Conv2DЙ
%ResNet9/conv2d/BiasAdd/ReadVariableOpReadVariableOp.resnet9_conv2d_biasadd_readvariableop_resource*
dtype0*
_output_shapes
: 2'
%ResNet9/conv2d/BiasAdd/ReadVariableOpФ
ResNet9/conv2d/BiasAddBiasAddResNet9/conv2d/Conv2D:output:0-ResNet9/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
ResNet9/conv2d/BiasAdd
(ResNet9/batch_normalization/LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: 2*
(ResNet9/batch_normalization/LogicalAnd/x
(ResNet9/batch_normalization/LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2*
(ResNet9/batch_normalization/LogicalAnd/yм
&ResNet9/batch_normalization/LogicalAnd
LogicalAnd1ResNet9/batch_normalization/LogicalAnd/x:output:01ResNet9/batch_normalization/LogicalAnd/y:output:0*
_output_shapes
: 2(
&ResNet9/batch_normalization/LogicalAndШ
*ResNet9/batch_normalization/ReadVariableOpReadVariableOp3resnet9_batch_normalization_readvariableop_resource*
dtype0*
_output_shapes
: 2,
*ResNet9/batch_normalization/ReadVariableOpЮ
,ResNet9/batch_normalization/ReadVariableOp_1ReadVariableOp5resnet9_batch_normalization_readvariableop_1_resource*
dtype0*
_output_shapes
: 2.
,ResNet9/batch_normalization/ReadVariableOp_1ћ
;ResNet9/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpDresnet9_batch_normalization_fusedbatchnormv3_readvariableop_resource*
dtype0*
_output_shapes
: 2=
;ResNet9/batch_normalization/FusedBatchNormV3/ReadVariableOp
=ResNet9/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFresnet9_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
dtype0*
_output_shapes
: 2?
=ResNet9/batch_normalization/FusedBatchNormV3/ReadVariableOp_1
,ResNet9/batch_normalization/FusedBatchNormV3FusedBatchNormV3ResNet9/conv2d/BiasAdd:output:02ResNet9/batch_normalization/ReadVariableOp:value:04ResNet9/batch_normalization/ReadVariableOp_1:value:0CResNet9/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0EResNet9/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
epsilon%o:*K
_output_shapes9
7:џџџџџџџџџ : : : : :*
T0*
U0*
is_training( 2.
,ResNet9/batch_normalization/FusedBatchNormV3
!ResNet9/batch_normalization/ConstConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2#
!ResNet9/batch_normalization/ConstЊ
ResNet9/activation_1/ReluRelu0ResNet9/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
ResNet9/activation_1/ReluШ
&ResNet9/conv2d_1/Conv2D/ReadVariableOpReadVariableOp/resnet9_conv2d_1_conv2d_readvariableop_resource*
dtype0*&
_output_shapes
:  2(
&ResNet9/conv2d_1/Conv2D/ReadVariableOpї
ResNet9/conv2d_1/Conv2DConv2D'ResNet9/activation_1/Relu:activations:0.ResNet9/conv2d_1/Conv2D/ReadVariableOp:value:0*/
_output_shapes
:џџџџџџџџџ *
T0*
strides
*
paddingSAME2
ResNet9/conv2d_1/Conv2DП
'ResNet9/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp0resnet9_conv2d_1_biasadd_readvariableop_resource*
dtype0*
_output_shapes
: 2)
'ResNet9/conv2d_1/BiasAdd/ReadVariableOpЬ
ResNet9/conv2d_1/BiasAddBiasAdd ResNet9/conv2d_1/Conv2D:output:0/ResNet9/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
ResNet9/conv2d_1/BiasAdd
*ResNet9/batch_normalization_1/LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: 2,
*ResNet9/batch_normalization_1/LogicalAnd/x
*ResNet9/batch_normalization_1/LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2,
*ResNet9/batch_normalization_1/LogicalAnd/yф
(ResNet9/batch_normalization_1/LogicalAnd
LogicalAnd3ResNet9/batch_normalization_1/LogicalAnd/x:output:03ResNet9/batch_normalization_1/LogicalAnd/y:output:0*
_output_shapes
: 2*
(ResNet9/batch_normalization_1/LogicalAndЮ
,ResNet9/batch_normalization_1/ReadVariableOpReadVariableOp5resnet9_batch_normalization_1_readvariableop_resource*
dtype0*
_output_shapes
: 2.
,ResNet9/batch_normalization_1/ReadVariableOpд
.ResNet9/batch_normalization_1/ReadVariableOp_1ReadVariableOp7resnet9_batch_normalization_1_readvariableop_1_resource*
dtype0*
_output_shapes
: 20
.ResNet9/batch_normalization_1/ReadVariableOp_1
=ResNet9/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpFresnet9_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
dtype0*
_output_shapes
: 2?
=ResNet9/batch_normalization_1/FusedBatchNormV3/ReadVariableOp
?ResNet9/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHresnet9_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
dtype0*
_output_shapes
: 2A
?ResNet9/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1
.ResNet9/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3!ResNet9/conv2d_1/BiasAdd:output:04ResNet9/batch_normalization_1/ReadVariableOp:value:06ResNet9/batch_normalization_1/ReadVariableOp_1:value:0EResNet9/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0GResNet9/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*
is_training( *
epsilon%o:*K
_output_shapes9
7:џџџџџџџџџ : : : : :20
.ResNet9/batch_normalization_1/FusedBatchNormV3
#ResNet9/batch_normalization_1/ConstConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2%
#ResNet9/batch_normalization_1/ConstЌ
ResNet9/activation_2/ReluRelu2ResNet9/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
ResNet9/activation_2/ReluЖ
ResNet9/add/addAddV2'ResNet9/activation_2/Relu:activations:0&ResNet9/max_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
ResNet9/add/add
ResNet9/activation_3/ReluReluResNet9/add/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
ResNet9/activation_3/ReluШ
&ResNet9/conv2d_3/Conv2D/ReadVariableOpReadVariableOp/resnet9_conv2d_3_conv2d_readvariableop_resource*
dtype0*&
_output_shapes
: @2(
&ResNet9/conv2d_3/Conv2D/ReadVariableOpї
ResNet9/conv2d_3/Conv2DConv2D'ResNet9/activation_3/Relu:activations:0.ResNet9/conv2d_3/Conv2D/ReadVariableOp:value:0*/
_output_shapes
:џџџџџџџџџ@*
T0*
strides
*
paddingSAME2
ResNet9/conv2d_3/Conv2DП
'ResNet9/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp0resnet9_conv2d_3_biasadd_readvariableop_resource*
dtype0*
_output_shapes
:@2)
'ResNet9/conv2d_3/BiasAdd/ReadVariableOpЬ
ResNet9/conv2d_3/BiasAddBiasAdd ResNet9/conv2d_3/Conv2D:output:0/ResNet9/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
ResNet9/conv2d_3/BiasAdd
*ResNet9/batch_normalization_3/LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: 2,
*ResNet9/batch_normalization_3/LogicalAnd/x
*ResNet9/batch_normalization_3/LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2,
*ResNet9/batch_normalization_3/LogicalAnd/yф
(ResNet9/batch_normalization_3/LogicalAnd
LogicalAnd3ResNet9/batch_normalization_3/LogicalAnd/x:output:03ResNet9/batch_normalization_3/LogicalAnd/y:output:0*
_output_shapes
: 2*
(ResNet9/batch_normalization_3/LogicalAndЮ
,ResNet9/batch_normalization_3/ReadVariableOpReadVariableOp5resnet9_batch_normalization_3_readvariableop_resource*
dtype0*
_output_shapes
:@2.
,ResNet9/batch_normalization_3/ReadVariableOpд
.ResNet9/batch_normalization_3/ReadVariableOp_1ReadVariableOp7resnet9_batch_normalization_3_readvariableop_1_resource*
dtype0*
_output_shapes
:@20
.ResNet9/batch_normalization_3/ReadVariableOp_1
=ResNet9/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpFresnet9_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
dtype0*
_output_shapes
:@2?
=ResNet9/batch_normalization_3/FusedBatchNormV3/ReadVariableOp
?ResNet9/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHresnet9_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
dtype0*
_output_shapes
:@2A
?ResNet9/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1
.ResNet9/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3!ResNet9/conv2d_3/BiasAdd:output:04ResNet9/batch_normalization_3/ReadVariableOp:value:06ResNet9/batch_normalization_3/ReadVariableOp_1:value:0EResNet9/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0GResNet9/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*
is_training( *
epsilon%o:*K
_output_shapes9
7:џџџџџџџџџ@:@:@:@:@:20
.ResNet9/batch_normalization_3/FusedBatchNormV3
#ResNet9/batch_normalization_3/ConstConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2%
#ResNet9/batch_normalization_3/ConstЌ
ResNet9/activation_4/ReluRelu2ResNet9/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
ResNet9/activation_4/ReluШ
&ResNet9/conv2d_4/Conv2D/ReadVariableOpReadVariableOp/resnet9_conv2d_4_conv2d_readvariableop_resource*
dtype0*&
_output_shapes
:@@2(
&ResNet9/conv2d_4/Conv2D/ReadVariableOpї
ResNet9/conv2d_4/Conv2DConv2D'ResNet9/activation_4/Relu:activations:0.ResNet9/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ@2
ResNet9/conv2d_4/Conv2DП
'ResNet9/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp0resnet9_conv2d_4_biasadd_readvariableop_resource*
dtype0*
_output_shapes
:@2)
'ResNet9/conv2d_4/BiasAdd/ReadVariableOpЬ
ResNet9/conv2d_4/BiasAddBiasAdd ResNet9/conv2d_4/Conv2D:output:0/ResNet9/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
ResNet9/conv2d_4/BiasAddШ
&ResNet9/conv2d_2/Conv2D/ReadVariableOpReadVariableOp/resnet9_conv2d_2_conv2d_readvariableop_resource*
dtype0*&
_output_shapes
: @2(
&ResNet9/conv2d_2/Conv2D/ReadVariableOpї
ResNet9/conv2d_2/Conv2DConv2D'ResNet9/activation_3/Relu:activations:0.ResNet9/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ@2
ResNet9/conv2d_2/Conv2DП
'ResNet9/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp0resnet9_conv2d_2_biasadd_readvariableop_resource*
dtype0*
_output_shapes
:@2)
'ResNet9/conv2d_2/BiasAdd/ReadVariableOpЬ
ResNet9/conv2d_2/BiasAddBiasAdd ResNet9/conv2d_2/Conv2D:output:0/ResNet9/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
ResNet9/conv2d_2/BiasAdd
*ResNet9/batch_normalization_4/LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: 2,
*ResNet9/batch_normalization_4/LogicalAnd/x
*ResNet9/batch_normalization_4/LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2,
*ResNet9/batch_normalization_4/LogicalAnd/yф
(ResNet9/batch_normalization_4/LogicalAnd
LogicalAnd3ResNet9/batch_normalization_4/LogicalAnd/x:output:03ResNet9/batch_normalization_4/LogicalAnd/y:output:0*
_output_shapes
: 2*
(ResNet9/batch_normalization_4/LogicalAndЮ
,ResNet9/batch_normalization_4/ReadVariableOpReadVariableOp5resnet9_batch_normalization_4_readvariableop_resource*
dtype0*
_output_shapes
:@2.
,ResNet9/batch_normalization_4/ReadVariableOpд
.ResNet9/batch_normalization_4/ReadVariableOp_1ReadVariableOp7resnet9_batch_normalization_4_readvariableop_1_resource*
dtype0*
_output_shapes
:@20
.ResNet9/batch_normalization_4/ReadVariableOp_1
=ResNet9/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpFresnet9_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
dtype0*
_output_shapes
:@2?
=ResNet9/batch_normalization_4/FusedBatchNormV3/ReadVariableOp
?ResNet9/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHresnet9_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
dtype0*
_output_shapes
:@2A
?ResNet9/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1
.ResNet9/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3!ResNet9/conv2d_4/BiasAdd:output:04ResNet9/batch_normalization_4/ReadVariableOp:value:06ResNet9/batch_normalization_4/ReadVariableOp_1:value:0EResNet9/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0GResNet9/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*
is_training( *
epsilon%o:*K
_output_shapes9
7:џџџџџџџџџ@:@:@:@:@:20
.ResNet9/batch_normalization_4/FusedBatchNormV3
#ResNet9/batch_normalization_4/ConstConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2%
#ResNet9/batch_normalization_4/ConstЌ
ResNet9/activation_5/ReluRelu2ResNet9/batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
ResNet9/activation_5/Relu
*ResNet9/batch_normalization_2/LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: 2,
*ResNet9/batch_normalization_2/LogicalAnd/x
*ResNet9/batch_normalization_2/LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2,
*ResNet9/batch_normalization_2/LogicalAnd/yф
(ResNet9/batch_normalization_2/LogicalAnd
LogicalAnd3ResNet9/batch_normalization_2/LogicalAnd/x:output:03ResNet9/batch_normalization_2/LogicalAnd/y:output:0*
_output_shapes
: 2*
(ResNet9/batch_normalization_2/LogicalAndЮ
,ResNet9/batch_normalization_2/ReadVariableOpReadVariableOp5resnet9_batch_normalization_2_readvariableop_resource*
dtype0*
_output_shapes
:@2.
,ResNet9/batch_normalization_2/ReadVariableOpд
.ResNet9/batch_normalization_2/ReadVariableOp_1ReadVariableOp7resnet9_batch_normalization_2_readvariableop_1_resource*
dtype0*
_output_shapes
:@20
.ResNet9/batch_normalization_2/ReadVariableOp_1
=ResNet9/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpFresnet9_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
dtype0*
_output_shapes
:@2?
=ResNet9/batch_normalization_2/FusedBatchNormV3/ReadVariableOp
?ResNet9/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHresnet9_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
dtype0*
_output_shapes
:@2A
?ResNet9/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1
.ResNet9/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3!ResNet9/conv2d_2/BiasAdd:output:04ResNet9/batch_normalization_2/ReadVariableOp:value:06ResNet9/batch_normalization_2/ReadVariableOp_1:value:0EResNet9/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0GResNet9/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*
is_training( *
epsilon%o:*K
_output_shapes9
7:џџџџџџџџџ@:@:@:@:@:20
.ResNet9/batch_normalization_2/FusedBatchNormV3
#ResNet9/batch_normalization_2/ConstConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2%
#ResNet9/batch_normalization_2/ConstЦ
ResNet9/add_1/addAddV2'ResNet9/activation_5/Relu:activations:02ResNet9/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
ResNet9/add_1/add
ResNet9/activation_6/ReluReluResNet9/add_1/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
ResNet9/activation_6/ReluУ
7ResNet9/global_average_pooling2d/Mean/reduction_indicesConst*
valueB"      *
dtype0*
_output_shapes
:29
7ResNet9/global_average_pooling2d/Mean/reduction_indicesѓ
%ResNet9/global_average_pooling2d/MeanMean'ResNet9/activation_6/Relu:activations:0@ResNet9/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2'
%ResNet9/global_average_pooling2d/Mean
ResNet9/flatten/ConstConst*
valueB"џџџџ@   *
dtype0*
_output_shapes
:2
ResNet9/flatten/ConstП
ResNet9/flatten/ReshapeReshape.ResNet9/global_average_pooling2d/Mean:output:0ResNet9/flatten/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
ResNet9/flatten/ReshapeЎ
 ResNet9/fc/MatMul/ReadVariableOpReadVariableOp)resnet9_fc_matmul_readvariableop_resource*
dtype0*
_output_shapes

:@2"
 ResNet9/fc/MatMul/ReadVariableOpЎ
ResNet9/fc/MatMulMatMul ResNet9/flatten/Reshape:output:0(ResNet9/fc/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ResNet9/fc/MatMul­
!ResNet9/fc/BiasAdd/ReadVariableOpReadVariableOp*resnet9_fc_biasadd_readvariableop_resource*
dtype0*
_output_shapes
:2#
!ResNet9/fc/BiasAdd/ReadVariableOp­
ResNet9/fc/BiasAddBiasAddResNet9/fc/MatMul:product:0)ResNet9/fc/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ResNet9/fc/BiasAdd
ResNet9/fc/SigmoidSigmoidResNet9/fc/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ResNet9/fc/SigmoidЉ
IdentityIdentityResNet9/fc/Sigmoid:y:0<^ResNet9/batch_normalization/FusedBatchNormV3/ReadVariableOp>^ResNet9/batch_normalization/FusedBatchNormV3/ReadVariableOp_1+^ResNet9/batch_normalization/ReadVariableOp-^ResNet9/batch_normalization/ReadVariableOp_1>^ResNet9/batch_normalization_1/FusedBatchNormV3/ReadVariableOp@^ResNet9/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1-^ResNet9/batch_normalization_1/ReadVariableOp/^ResNet9/batch_normalization_1/ReadVariableOp_1>^ResNet9/batch_normalization_2/FusedBatchNormV3/ReadVariableOp@^ResNet9/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1-^ResNet9/batch_normalization_2/ReadVariableOp/^ResNet9/batch_normalization_2/ReadVariableOp_1>^ResNet9/batch_normalization_3/FusedBatchNormV3/ReadVariableOp@^ResNet9/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1-^ResNet9/batch_normalization_3/ReadVariableOp/^ResNet9/batch_normalization_3/ReadVariableOp_1>^ResNet9/batch_normalization_4/FusedBatchNormV3/ReadVariableOp@^ResNet9/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1-^ResNet9/batch_normalization_4/ReadVariableOp/^ResNet9/batch_normalization_4/ReadVariableOp_11^ResNet9/bn_conv1/FusedBatchNormV3/ReadVariableOp3^ResNet9/bn_conv1/FusedBatchNormV3/ReadVariableOp_1 ^ResNet9/bn_conv1/ReadVariableOp"^ResNet9/bn_conv1/ReadVariableOp_1%^ResNet9/conv1/BiasAdd/ReadVariableOp$^ResNet9/conv1/Conv2D/ReadVariableOp&^ResNet9/conv2d/BiasAdd/ReadVariableOp%^ResNet9/conv2d/Conv2D/ReadVariableOp(^ResNet9/conv2d_1/BiasAdd/ReadVariableOp'^ResNet9/conv2d_1/Conv2D/ReadVariableOp(^ResNet9/conv2d_2/BiasAdd/ReadVariableOp'^ResNet9/conv2d_2/Conv2D/ReadVariableOp(^ResNet9/conv2d_3/BiasAdd/ReadVariableOp'^ResNet9/conv2d_3/Conv2D/ReadVariableOp(^ResNet9/conv2d_4/BiasAdd/ReadVariableOp'^ResNet9/conv2d_4/Conv2D/ReadVariableOp"^ResNet9/fc/BiasAdd/ReadVariableOp!^ResNet9/fc/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*Ш
_input_shapesЖ
Г:џџџџџџџџџ		::::::::::::::::::::::::::::::::::::::2z
;ResNet9/batch_normalization/FusedBatchNormV3/ReadVariableOp;ResNet9/batch_normalization/FusedBatchNormV3/ReadVariableOp2\
,ResNet9/batch_normalization_2/ReadVariableOp,ResNet9/batch_normalization_2/ReadVariableOp2
?ResNet9/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?ResNet9/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12`
.ResNet9/batch_normalization_3/ReadVariableOp_1.ResNet9/batch_normalization_3/ReadVariableOp_12\
,ResNet9/batch_normalization/ReadVariableOp_1,ResNet9/batch_normalization/ReadVariableOp_12D
 ResNet9/fc/MatMul/ReadVariableOp ResNet9/fc/MatMul/ReadVariableOp2`
.ResNet9/batch_normalization_4/ReadVariableOp_1.ResNet9/batch_normalization_4/ReadVariableOp_12R
'ResNet9/conv2d_3/BiasAdd/ReadVariableOp'ResNet9/conv2d_3/BiasAdd/ReadVariableOp2
?ResNet9/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?ResNet9/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12\
,ResNet9/batch_normalization_3/ReadVariableOp,ResNet9/batch_normalization_3/ReadVariableOp2L
$ResNet9/conv1/BiasAdd/ReadVariableOp$ResNet9/conv1/BiasAdd/ReadVariableOp2
?ResNet9/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?ResNet9/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12J
#ResNet9/conv1/Conv2D/ReadVariableOp#ResNet9/conv1/Conv2D/ReadVariableOp2\
,ResNet9/batch_normalization_4/ReadVariableOp,ResNet9/batch_normalization_4/ReadVariableOp2R
'ResNet9/conv2d_2/BiasAdd/ReadVariableOp'ResNet9/conv2d_2/BiasAdd/ReadVariableOp2P
&ResNet9/conv2d_1/Conv2D/ReadVariableOp&ResNet9/conv2d_1/Conv2D/ReadVariableOp2h
2ResNet9/bn_conv1/FusedBatchNormV3/ReadVariableOp_12ResNet9/bn_conv1/FusedBatchNormV3/ReadVariableOp_12X
*ResNet9/batch_normalization/ReadVariableOp*ResNet9/batch_normalization/ReadVariableOp2F
!ResNet9/fc/BiasAdd/ReadVariableOp!ResNet9/fc/BiasAdd/ReadVariableOp2~
=ResNet9/batch_normalization_1/FusedBatchNormV3/ReadVariableOp=ResNet9/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2~
=ResNet9/batch_normalization_2/FusedBatchNormV3/ReadVariableOp=ResNet9/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2~
=ResNet9/batch_normalization_3/FusedBatchNormV3/ReadVariableOp=ResNet9/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2P
&ResNet9/conv2d_2/Conv2D/ReadVariableOp&ResNet9/conv2d_2/Conv2D/ReadVariableOp2~
=ResNet9/batch_normalization_4/FusedBatchNormV3/ReadVariableOp=ResNet9/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2B
ResNet9/bn_conv1/ReadVariableOpResNet9/bn_conv1/ReadVariableOp2R
'ResNet9/conv2d_1/BiasAdd/ReadVariableOp'ResNet9/conv2d_1/BiasAdd/ReadVariableOp2P
&ResNet9/conv2d_3/Conv2D/ReadVariableOp&ResNet9/conv2d_3/Conv2D/ReadVariableOp2
?ResNet9/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?ResNet9/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12d
0ResNet9/bn_conv1/FusedBatchNormV3/ReadVariableOp0ResNet9/bn_conv1/FusedBatchNormV3/ReadVariableOp2\
,ResNet9/batch_normalization_1/ReadVariableOp,ResNet9/batch_normalization_1/ReadVariableOp2R
'ResNet9/conv2d_4/BiasAdd/ReadVariableOp'ResNet9/conv2d_4/BiasAdd/ReadVariableOp2`
.ResNet9/batch_normalization_1/ReadVariableOp_1.ResNet9/batch_normalization_1/ReadVariableOp_12P
&ResNet9/conv2d_4/Conv2D/ReadVariableOp&ResNet9/conv2d_4/Conv2D/ReadVariableOp2N
%ResNet9/conv2d/BiasAdd/ReadVariableOp%ResNet9/conv2d/BiasAdd/ReadVariableOp2L
$ResNet9/conv2d/Conv2D/ReadVariableOp$ResNet9/conv2d/Conv2D/ReadVariableOp2~
=ResNet9/batch_normalization/FusedBatchNormV3/ReadVariableOp_1=ResNet9/batch_normalization/FusedBatchNormV3/ReadVariableOp_12`
.ResNet9/batch_normalization_2/ReadVariableOp_1.ResNet9/batch_normalization_2/ReadVariableOp_12F
!ResNet9/bn_conv1/ReadVariableOp_1!ResNet9/bn_conv1/ReadVariableOp_1:' #
!
_user_specified_name	input_1
Р	
ж
=__inference_fc_layer_call_and_return_conditional_losses_26294

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
dtype0*
_output_shapes

:@2
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
dtype0*
_output_shapes
:2
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs
ъ
ў
5__inference_batch_normalization_2_layer_call_fn_26241

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityЂStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_23602*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin	
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*,
_gradient_op_typePartitionedCallUnused2
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs


м
C__inference_conv2d_4_layer_call_and_return_conditional_losses_23349

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*
dtype0*&
_output_shapes
:@@2
Conv2D/ReadVariableOpЕ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
paddingSAME*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
T0*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
dtype0*
_output_shapes
:@2
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2	
BiasAddЏ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
T02

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
Д
ў
5__inference_batch_normalization_1_layer_call_fn_25635

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityЂStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*,
_gradient_op_typePartitionedCallUnused*Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_23884*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:џџџџџџџџџ *
Tin	
22
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ъ
ў
5__inference_batch_normalization_4_layer_call_fn_26006

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityЂStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
Tin	
2*,
_gradient_op_typePartitionedCallUnused*Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_23482*
Tout
2**
config_proto

CPU

GPU 2J 82
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs

ё
(__inference_bn_conv1_layer_call_fn_25378

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityЂStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:џџџџџџџџџ *
Tin	
2*,
_gradient_op_typePartitionedCallUnused*L
fGRE
C__inference_bn_conv1_layer_call_and_return_conditional_losses_23715*
Tout
22
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*/
_output_shapes
:џџџџџџџџџ *
T02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Ъ
ѓ
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_25892

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: 2
LogicalAnd/x^
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
:@2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
:@2
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
dtype0*
_output_shapes
:@2!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
dtype0*
_output_shapes
:@2#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
epsilon%o:*K
_output_shapes9
7:џџџџџџџџџ@:@:@:@:@:*
T0*
U0*
is_training( 2
FusedBatchNormV3S
ConstConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2
Constк
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*/
_output_shapes
:џџџџџџџџџ@*
T02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ@::::2 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_1:& "
 
_user_specified_nameinputs
Ъ
ѓ
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_26062

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: 2
LogicalAnd/x^
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
:@2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
:@2
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
dtype0*
_output_shapes
:@2!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
dtype0*
_output_shapes
:@2#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
is_training( *
epsilon%o:*K
_output_shapes9
7:џџџџџџџџџ@:@:@:@:@:*
T0*
U02
FusedBatchNormV3S
ConstConst*
dtype0*
_output_shapes
: *
valueB
 *Єp}?2
Constк
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*/
_output_shapes
:џџџџџџџџџ@*
T02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ@::::2 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_1:& "
 
_user_specified_nameinputs
ё
H
,__inference_activation_4_layer_call_fn_25920

inputs
identityЗ
PartitionedCallPartitionedCallinputs**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ@*,
_gradient_op_typePartitionedCallUnused*P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_24058*
Tout
22
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*/
_output_shapes
:џџџџџџџџџ@*
T02

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:& "
 
_user_specified_nameinputs
Є
c
G__inference_activation_1_layer_call_and_return_conditional_losses_25553

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ :& "
 
_user_specified_nameinputs
э
F
*__inference_activation_layer_call_fn_25388

inputs
identityЕ
PartitionedCallPartitionedCallinputs*
Tin
2*/
_output_shapes
:џџџџџџџџџ *,
_gradient_op_typePartitionedCallUnused*N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_23744*
Tout
2**
config_proto

CPU

GPU 2J 82
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ :& "
 
_user_specified_nameinputs
О
Љ
(__inference_conv2d_4_layer_call_fn_23357

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_23349*
Tout
2**
config_proto

CPU

GPU 2J 8*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
Tin
2*,
_gradient_op_typePartitionedCallUnused2
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ъ
ў
5__inference_batch_normalization_1_layer_call_fn_25709

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityЂStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_23149*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin	
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *,
_gradient_op_typePartitionedCallUnused2
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
T02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
§	
й
@__inference_conv1_layer_call_and_return_conditional_losses_22733

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*
dtype0*&
_output_shapes
: 2
Conv2D/ReadVariableOpЕ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
dtype0*
_output_shapes
: 2
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2	
BiasAddЏ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
T02

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
њ
ж
'__inference_ResNet9_layer_call_fn_24488
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32#
statefulpartitionedcall_args_33#
statefulpartitionedcall_args_34#
statefulpartitionedcall_args_35#
statefulpartitionedcall_args_36#
statefulpartitionedcall_args_37#
statefulpartitionedcall_args_38
identityЂStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32statefulpartitionedcall_args_33statefulpartitionedcall_args_34statefulpartitionedcall_args_35statefulpartitionedcall_args_36statefulpartitionedcall_args_37statefulpartitionedcall_args_38*2
Tin+
)2'*'
_output_shapes
:џџџџџџџџџ*,
_gradient_op_typePartitionedCallUnused*K
fFRD
B__inference_ResNet9_layer_call_and_return_conditional_losses_24447*
Tout
2**
config_proto

CPU

GPU 2J 82
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:џџџџџџџџџ*
T02

Identity"
identityIdentity:output:0*Ш
_input_shapesЖ
Г:џџџџџџџџџ		::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
Ъ
ѓ
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_26158

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
dtype0
*
_output_shapes
: *
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
:@2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
:@2
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
dtype0*
_output_shapes
:@2!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
dtype0*
_output_shapes
:@2#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*K
_output_shapes9
7:џџџџџџџџџ@:@:@:@:@:*
T0*
U0*
is_training( *
epsilon%o:2
FusedBatchNormV3S
ConstConst*
dtype0*
_output_shapes
: *
valueB
 *Єp}?2
Constк
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*/
_output_shapes
:џџџџџџџџџ@*
T02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ@::::2 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_1:& "
 
_user_specified_nameinputs
О
Љ
(__inference_conv2d_1_layer_call_fn_23055

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
Tin
2*,
_gradient_op_typePartitionedCallUnused*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_23047*
Tout
22
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ё
H
,__inference_activation_3_layer_call_fn_25750

inputs
identityЗ
PartitionedCallPartitionedCallinputs*,
_gradient_op_typePartitionedCallUnused*P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_23963*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:џџџџџџџџџ *
Tin
22
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ :& "
 
_user_specified_nameinputs
У
O
#__inference_add_layer_call_fn_25740
inputs_0
inputs_1
identityЛ
PartitionedCallPartitionedCallinputs_0inputs_1*,
_gradient_op_typePartitionedCallUnused*G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_23949*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ 2
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:џџџџџџџџџ :џџџџџџџџџ :($
"
_user_specified_name
inputs/1:( $
"
_user_specified_name
inputs/0
ц#

C__inference_bn_conv1_layer_call_and_return_conditional_losses_23693

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_23678
assignmovingavg_1_23685
identityЂ#AssignMovingAvg/AssignSubVariableOpЂAssignMovingAvg/ReadVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpЂ AssignMovingAvg_1/ReadVariableOpЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
dtype0
*
_output_shapes
: *
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
: 2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
: 2
ReadVariableOp_1Q
ConstConst*
dtype0*
_output_shapes
: *
valueB 2
ConstU
Const_1Const*
valueB *
dtype0*
_output_shapes
: 2	
Const_1
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*K
_output_shapes9
7:џџџџџџџџџ : : : : :*
T0*
U0*
epsilon%o:2
FusedBatchNormV3W
Const_2Const*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2	
Const_2
AssignMovingAvg/sub/xConst*
valueB
 *  ?*(
_class
loc:@AssignMovingAvg/23678*
dtype0*
_output_shapes
: 2
AssignMovingAvg/sub/xЎ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/23678*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_23678*
dtype0*
_output_shapes
: 2 
AssignMovingAvg/ReadVariableOpЫ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/23678*
_output_shapes
: 2
AssignMovingAvg/sub_1Д
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/23678*
_output_shapes
: 2
AssignMovingAvg/mulџ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_23678AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
dtype0*
_output_shapes
 *(
_class
loc:@AssignMovingAvg/236782%
#AssignMovingAvg/AssignSubVariableOpЃ
AssignMovingAvg_1/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?**
_class 
loc:@AssignMovingAvg_1/236852
AssignMovingAvg_1/sub/xЖ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
_output_shapes
: *
T0**
_class 
loc:@AssignMovingAvg_1/236852
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_23685*
dtype0*
_output_shapes
: 2"
 AssignMovingAvg_1/ReadVariableOpз
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
_output_shapes
: *
T0**
_class 
loc:@AssignMovingAvg_1/236852
AssignMovingAvg_1/sub_1О
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
_output_shapes
: *
T0**
_class 
loc:@AssignMovingAvg_1/236852
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_23685AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
dtype0*
_output_shapes
 **
_class 
loc:@AssignMovingAvg_1/236852'
%AssignMovingAvg_1/AssignSubVariableOpІ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ ::::2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_12@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp:& "
 
_user_specified_nameinputs
А
ќ
3__inference_batch_normalization_layer_call_fn_25539

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityЂStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_23789*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin	
2*/
_output_shapes
:џџџџџџџџџ *,
_gradient_op_typePartitionedCallUnused2
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ї
е
'__inference_ResNet9_layer_call_fn_25218

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32#
statefulpartitionedcall_args_33#
statefulpartitionedcall_args_34#
statefulpartitionedcall_args_35#
statefulpartitionedcall_args_36#
statefulpartitionedcall_args_37#
statefulpartitionedcall_args_38
identityЂStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32statefulpartitionedcall_args_33statefulpartitionedcall_args_34statefulpartitionedcall_args_35statefulpartitionedcall_args_36statefulpartitionedcall_args_37statefulpartitionedcall_args_38*2
Tin+
)2'*'
_output_shapes
:џџџџџџџџџ*,
_gradient_op_typePartitionedCallUnused*K
fFRD
B__inference_ResNet9_layer_call_and_return_conditional_losses_24557*
Tout
2**
config_proto

CPU

GPU 2J 82
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*Ш
_input_shapesЖ
Г:џџџџџџџџџ		::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ё
H
,__inference_activation_6_layer_call_fn_26272

inputs
identityЗ
PartitionedCallPartitionedCallinputs**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ@*,
_gradient_op_typePartitionedCallUnused*P
fKRI
G__inference_activation_6_layer_call_and_return_conditional_losses_24263*
Tout
22
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:& "
 
_user_specified_nameinputs
Є
c
G__inference_activation_6_layer_call_and_return_conditional_losses_24263

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:& "
 
_user_specified_nameinputs
д
в
#__inference_signature_wrapper_24722
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32#
statefulpartitionedcall_args_33#
statefulpartitionedcall_args_34#
statefulpartitionedcall_args_35#
statefulpartitionedcall_args_36#
statefulpartitionedcall_args_37#
statefulpartitionedcall_args_38
identityЂStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32statefulpartitionedcall_args_33statefulpartitionedcall_args_34statefulpartitionedcall_args_35statefulpartitionedcall_args_36statefulpartitionedcall_args_37statefulpartitionedcall_args_38*2
Tin+
)2'*'
_output_shapes
:џџџџџџџџџ*,
_gradient_op_typePartitionedCallUnused*)
f$R"
 __inference__wrapped_model_22722*
Tout
2**
config_proto

CPU

GPU 2J 82
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*Ш
_input_shapesЖ
Г:џџџџџџџџџ		::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
є
l
@__inference_add_1_layer_call_and_return_conditional_losses_26256
inputs_0
inputs_1
identitya
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:џџџџџџџџџ@2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:џџџџџџџџџ@:џџџџџџџџџ@:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1


м
C__inference_conv2d_3_layer_call_and_return_conditional_losses_23198

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*
dtype0*&
_output_shapes
: @2
Conv2D/ReadVariableOpЕ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
dtype0*
_output_shapes
:@2
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
T02	
BiasAddЏ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
T02

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
ђ
j
>__inference_add_layer_call_and_return_conditional_losses_25734
inputs_0
inputs_1
identitya
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:џџџџџџџџџ 2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:џџџџџџџџџ :џџџџџџџџџ :( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
Ш
ё
N__inference_batch_normalization_layer_call_and_return_conditional_losses_23811

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
dtype0
*
_output_shapes
: *
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
: 2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
: 2
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
dtype0*
_output_shapes
: 2!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
dtype0*
_output_shapes
: 2#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
epsilon%o:*K
_output_shapes9
7:џџџџџџџџџ : : : : :*
T0*
U0*
is_training( 2
FusedBatchNormV3S
ConstConst*
dtype0*
_output_shapes
: *
valueB
 *Єp}?2
Constк
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*/
_output_shapes
:џџџџџџџџџ *
T02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ ::::2 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_1:& "
 
_user_specified_nameinputs
э
o
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_23647

inputs
identity
Mean/reduction_indicesConst*
valueB"      *
dtype0*
_output_shapes
:2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Meanj
IdentityIdentityMean:output:0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T02

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:& "
 
_user_specified_nameinputs
љ
 
B__inference_ResNet9_layer_call_and_return_conditional_losses_24963

inputs(
$conv1_conv2d_readvariableop_resource)
%conv1_biasadd_readvariableop_resource$
 bn_conv1_readvariableop_resource&
"bn_conv1_readvariableop_1_resource"
bn_conv1_assignmovingavg_24747$
 bn_conv1_assignmovingavg_1_24754)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource-
)batch_normalization_assignmovingavg_24785/
+batch_normalization_assignmovingavg_1_24792+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resource/
+batch_normalization_1_assignmovingavg_248221
-batch_normalization_1_assignmovingavg_1_24829+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resource/
+batch_normalization_3_assignmovingavg_248611
-batch_normalization_3_assignmovingavg_1_24868+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resource/
+batch_normalization_4_assignmovingavg_249041
-batch_normalization_4_assignmovingavg_1_249111
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resource/
+batch_normalization_2_assignmovingavg_249351
-batch_normalization_2_assignmovingavg_1_24942%
!fc_matmul_readvariableop_resource&
"fc_biasadd_readvariableop_resource
identityЂ7batch_normalization/AssignMovingAvg/AssignSubVariableOpЂ2batch_normalization/AssignMovingAvg/ReadVariableOpЂ9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpЂ4batch_normalization/AssignMovingAvg_1/ReadVariableOpЂ"batch_normalization/ReadVariableOpЂ$batch_normalization/ReadVariableOp_1Ђ9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpЂ4batch_normalization_1/AssignMovingAvg/ReadVariableOpЂ;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpЂ6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpЂ$batch_normalization_1/ReadVariableOpЂ&batch_normalization_1/ReadVariableOp_1Ђ9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpЂ4batch_normalization_2/AssignMovingAvg/ReadVariableOpЂ;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpЂ6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpЂ$batch_normalization_2/ReadVariableOpЂ&batch_normalization_2/ReadVariableOp_1Ђ9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpЂ4batch_normalization_3/AssignMovingAvg/ReadVariableOpЂ;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpЂ6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpЂ$batch_normalization_3/ReadVariableOpЂ&batch_normalization_3/ReadVariableOp_1Ђ9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpЂ4batch_normalization_4/AssignMovingAvg/ReadVariableOpЂ;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpЂ6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpЂ$batch_normalization_4/ReadVariableOpЂ&batch_normalization_4/ReadVariableOp_1Ђ,bn_conv1/AssignMovingAvg/AssignSubVariableOpЂ'bn_conv1/AssignMovingAvg/ReadVariableOpЂ.bn_conv1/AssignMovingAvg_1/AssignSubVariableOpЂ)bn_conv1/AssignMovingAvg_1/ReadVariableOpЂbn_conv1/ReadVariableOpЂbn_conv1/ReadVariableOp_1Ђconv1/BiasAdd/ReadVariableOpЂconv1/Conv2D/ReadVariableOpЂconv2d/BiasAdd/ReadVariableOpЂconv2d/Conv2D/ReadVariableOpЂconv2d_1/BiasAdd/ReadVariableOpЂconv2d_1/Conv2D/ReadVariableOpЂconv2d_2/BiasAdd/ReadVariableOpЂconv2d_2/Conv2D/ReadVariableOpЂconv2d_3/BiasAdd/ReadVariableOpЂconv2d_3/Conv2D/ReadVariableOpЂconv2d_4/BiasAdd/ReadVariableOpЂconv2d_4/Conv2D/ReadVariableOpЂfc/BiasAdd/ReadVariableOpЂfc/MatMul/ReadVariableOpЇ
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*
dtype0*&
_output_shapes
: 2
conv1/Conv2D/ReadVariableOpЕ
conv1/Conv2DConv2Dinputs#conv1/Conv2D/ReadVariableOp:value:0*
paddingSAME*/
_output_shapes
:џџџџџџџџџ *
T0*
strides
2
conv1/Conv2D
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
dtype0*
_output_shapes
: 2
conv1/BiasAdd/ReadVariableOp 
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:џџџџџџџџџ *
T02
conv1/BiasAddp
bn_conv1/LogicalAnd/xConst*
dtype0
*
_output_shapes
: *
value	B
 Z2
bn_conv1/LogicalAnd/xp
bn_conv1/LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
bn_conv1/LogicalAnd/y
bn_conv1/LogicalAnd
LogicalAndbn_conv1/LogicalAnd/x:output:0bn_conv1/LogicalAnd/y:output:0*
_output_shapes
: 2
bn_conv1/LogicalAnd
bn_conv1/ReadVariableOpReadVariableOp bn_conv1_readvariableop_resource*
dtype0*
_output_shapes
: 2
bn_conv1/ReadVariableOp
bn_conv1/ReadVariableOp_1ReadVariableOp"bn_conv1_readvariableop_1_resource*
dtype0*
_output_shapes
: 2
bn_conv1/ReadVariableOp_1c
bn_conv1/ConstConst*
dtype0*
_output_shapes
: *
valueB 2
bn_conv1/Constg
bn_conv1/Const_1Const*
valueB *
dtype0*
_output_shapes
: 2
bn_conv1/Const_1Ы
bn_conv1/FusedBatchNormV3FusedBatchNormV3conv1/BiasAdd:output:0bn_conv1/ReadVariableOp:value:0!bn_conv1/ReadVariableOp_1:value:0bn_conv1/Const:output:0bn_conv1/Const_1:output:0*
epsilon%o:*K
_output_shapes9
7:џџџџџџџџџ : : : : :*
T0*
U02
bn_conv1/FusedBatchNormV3i
bn_conv1/Const_2Const*
dtype0*
_output_shapes
: *
valueB
 *Єp}?2
bn_conv1/Const_2И
bn_conv1/AssignMovingAvg/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?*1
_class'
%#loc:@bn_conv1/AssignMovingAvg/247472 
bn_conv1/AssignMovingAvg/sub/xл
bn_conv1/AssignMovingAvg/subSub'bn_conv1/AssignMovingAvg/sub/x:output:0bn_conv1/Const_2:output:0*
_output_shapes
: *
T0*1
_class'
%#loc:@bn_conv1/AssignMovingAvg/247472
bn_conv1/AssignMovingAvg/sub­
'bn_conv1/AssignMovingAvg/ReadVariableOpReadVariableOpbn_conv1_assignmovingavg_24747*
dtype0*
_output_shapes
: 2)
'bn_conv1/AssignMovingAvg/ReadVariableOpј
bn_conv1/AssignMovingAvg/sub_1Sub/bn_conv1/AssignMovingAvg/ReadVariableOp:value:0&bn_conv1/FusedBatchNormV3:batch_mean:0*
_output_shapes
: *
T0*1
_class'
%#loc:@bn_conv1/AssignMovingAvg/247472 
bn_conv1/AssignMovingAvg/sub_1с
bn_conv1/AssignMovingAvg/mulMul"bn_conv1/AssignMovingAvg/sub_1:z:0 bn_conv1/AssignMovingAvg/sub:z:0*
T0*1
_class'
%#loc:@bn_conv1/AssignMovingAvg/24747*
_output_shapes
: 2
bn_conv1/AssignMovingAvg/mulЕ
,bn_conv1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpbn_conv1_assignmovingavg_24747 bn_conv1/AssignMovingAvg/mul:z:0(^bn_conv1/AssignMovingAvg/ReadVariableOp*
dtype0*
_output_shapes
 *1
_class'
%#loc:@bn_conv1/AssignMovingAvg/247472.
,bn_conv1/AssignMovingAvg/AssignSubVariableOpО
 bn_conv1/AssignMovingAvg_1/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?*3
_class)
'%loc:@bn_conv1/AssignMovingAvg_1/247542"
 bn_conv1/AssignMovingAvg_1/sub/xу
bn_conv1/AssignMovingAvg_1/subSub)bn_conv1/AssignMovingAvg_1/sub/x:output:0bn_conv1/Const_2:output:0*
_output_shapes
: *
T0*3
_class)
'%loc:@bn_conv1/AssignMovingAvg_1/247542 
bn_conv1/AssignMovingAvg_1/subГ
)bn_conv1/AssignMovingAvg_1/ReadVariableOpReadVariableOp bn_conv1_assignmovingavg_1_24754*
dtype0*
_output_shapes
: 2+
)bn_conv1/AssignMovingAvg_1/ReadVariableOp
 bn_conv1/AssignMovingAvg_1/sub_1Sub1bn_conv1/AssignMovingAvg_1/ReadVariableOp:value:0*bn_conv1/FusedBatchNormV3:batch_variance:0*
_output_shapes
: *
T0*3
_class)
'%loc:@bn_conv1/AssignMovingAvg_1/247542"
 bn_conv1/AssignMovingAvg_1/sub_1ы
bn_conv1/AssignMovingAvg_1/mulMul$bn_conv1/AssignMovingAvg_1/sub_1:z:0"bn_conv1/AssignMovingAvg_1/sub:z:0*
_output_shapes
: *
T0*3
_class)
'%loc:@bn_conv1/AssignMovingAvg_1/247542 
bn_conv1/AssignMovingAvg_1/mulС
.bn_conv1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp bn_conv1_assignmovingavg_1_24754"bn_conv1/AssignMovingAvg_1/mul:z:0*^bn_conv1/AssignMovingAvg_1/ReadVariableOp*
dtype0*
_output_shapes
 *3
_class)
'%loc:@bn_conv1/AssignMovingAvg_1/2475420
.bn_conv1/AssignMovingAvg_1/AssignSubVariableOp
activation/ReluRelubn_conv1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
activation/ReluФ
max_pooling2d/MaxPoolMaxPoolactivation/Relu:activations:0*
ksize
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ *
strides
2
max_pooling2d/MaxPoolЊ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*
dtype0*&
_output_shapes
:  2
conv2d/Conv2D/ReadVariableOpа
conv2d/Conv2DConv2Dmax_pooling2d/MaxPool:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ 2
conv2d/Conv2DЁ
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
dtype0*
_output_shapes
: 2
conv2d/BiasAdd/ReadVariableOpЄ
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:џџџџџџџџџ *
T02
conv2d/BiasAdd
 batch_normalization/LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2"
 batch_normalization/LogicalAnd/x
 batch_normalization/LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2"
 batch_normalization/LogicalAnd/yМ
batch_normalization/LogicalAnd
LogicalAnd)batch_normalization/LogicalAnd/x:output:0)batch_normalization/LogicalAnd/y:output:0*
_output_shapes
: 2 
batch_normalization/LogicalAndА
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
dtype0*
_output_shapes
: 2$
"batch_normalization/ReadVariableOpЖ
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
dtype0*
_output_shapes
: 2&
$batch_normalization/ReadVariableOp_1y
batch_normalization/ConstConst*
valueB *
dtype0*
_output_shapes
: 2
batch_normalization/Const}
batch_normalization/Const_1Const*
dtype0*
_output_shapes
: *
valueB 2
batch_normalization/Const_1
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0"batch_normalization/Const:output:0$batch_normalization/Const_1:output:0*K
_output_shapes9
7:џџџџџџџџџ : : : : :*
T0*
U0*
epsilon%o:2&
$batch_normalization/FusedBatchNormV3
batch_normalization/Const_2Const*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2
batch_normalization/Const_2й
)batch_normalization/AssignMovingAvg/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?*<
_class2
0.loc:@batch_normalization/AssignMovingAvg/247852+
)batch_normalization/AssignMovingAvg/sub/x
'batch_normalization/AssignMovingAvg/subSub2batch_normalization/AssignMovingAvg/sub/x:output:0$batch_normalization/Const_2:output:0*
T0*<
_class2
0.loc:@batch_normalization/AssignMovingAvg/24785*
_output_shapes
: 2)
'batch_normalization/AssignMovingAvg/subЮ
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp)batch_normalization_assignmovingavg_24785*
dtype0*
_output_shapes
: 24
2batch_normalization/AssignMovingAvg/ReadVariableOpЏ
)batch_normalization/AssignMovingAvg/sub_1Sub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:01batch_normalization/FusedBatchNormV3:batch_mean:0*
T0*<
_class2
0.loc:@batch_normalization/AssignMovingAvg/24785*
_output_shapes
: 2+
)batch_normalization/AssignMovingAvg/sub_1
'batch_normalization/AssignMovingAvg/mulMul-batch_normalization/AssignMovingAvg/sub_1:z:0+batch_normalization/AssignMovingAvg/sub:z:0*
_output_shapes
: *
T0*<
_class2
0.loc:@batch_normalization/AssignMovingAvg/247852)
'batch_normalization/AssignMovingAvg/mulї
7batch_normalization/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp)batch_normalization_assignmovingavg_24785+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*<
_class2
0.loc:@batch_normalization/AssignMovingAvg/24785*
dtype0*
_output_shapes
 29
7batch_normalization/AssignMovingAvg/AssignSubVariableOpп
+batch_normalization/AssignMovingAvg_1/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?*>
_class4
20loc:@batch_normalization/AssignMovingAvg_1/247922-
+batch_normalization/AssignMovingAvg_1/sub/x
)batch_normalization/AssignMovingAvg_1/subSub4batch_normalization/AssignMovingAvg_1/sub/x:output:0$batch_normalization/Const_2:output:0*
_output_shapes
: *
T0*>
_class4
20loc:@batch_normalization/AssignMovingAvg_1/247922+
)batch_normalization/AssignMovingAvg_1/subд
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp+batch_normalization_assignmovingavg_1_24792*
dtype0*
_output_shapes
: 26
4batch_normalization/AssignMovingAvg_1/ReadVariableOpЛ
+batch_normalization/AssignMovingAvg_1/sub_1Sub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:05batch_normalization/FusedBatchNormV3:batch_variance:0*
_output_shapes
: *
T0*>
_class4
20loc:@batch_normalization/AssignMovingAvg_1/247922-
+batch_normalization/AssignMovingAvg_1/sub_1Ђ
)batch_normalization/AssignMovingAvg_1/mulMul/batch_normalization/AssignMovingAvg_1/sub_1:z:0-batch_normalization/AssignMovingAvg_1/sub:z:0*
T0*>
_class4
20loc:@batch_normalization/AssignMovingAvg_1/24792*
_output_shapes
: 2+
)batch_normalization/AssignMovingAvg_1/mul
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp+batch_normalization_assignmovingavg_1_24792-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*
dtype0*
_output_shapes
 *>
_class4
20loc:@batch_normalization/AssignMovingAvg_1/247922;
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp
activation_1/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
activation_1/ReluА
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*
dtype0*&
_output_shapes
:  2 
conv2d_1/Conv2D/ReadVariableOpз
conv2d_1/Conv2DConv2Dactivation_1/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ 2
conv2d_1/Conv2DЇ
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
dtype0*
_output_shapes
: 2!
conv2d_1/BiasAdd/ReadVariableOpЌ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv2d_1/BiasAdd
"batch_normalization_1/LogicalAnd/xConst*
dtype0
*
_output_shapes
: *
value	B
 Z2$
"batch_normalization_1/LogicalAnd/x
"batch_normalization_1/LogicalAnd/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z2$
"batch_normalization_1/LogicalAnd/yФ
 batch_normalization_1/LogicalAnd
LogicalAnd+batch_normalization_1/LogicalAnd/x:output:0+batch_normalization_1/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_1/LogicalAndЖ
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
dtype0*
_output_shapes
: 2&
$batch_normalization_1/ReadVariableOpМ
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
dtype0*
_output_shapes
: 2(
&batch_normalization_1/ReadVariableOp_1}
batch_normalization_1/ConstConst*
dtype0*
_output_shapes
: *
valueB 2
batch_normalization_1/Const
batch_normalization_1/Const_1Const*
dtype0*
_output_shapes
: *
valueB 2
batch_normalization_1/Const_1
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0$batch_normalization_1/Const:output:0&batch_normalization_1/Const_1:output:0*K
_output_shapes9
7:џџџџџџџџџ : : : : :*
T0*
U0*
epsilon%o:2(
&batch_normalization_1/FusedBatchNormV3
batch_normalization_1/Const_2Const*
dtype0*
_output_shapes
: *
valueB
 *Єp}?2
batch_normalization_1/Const_2п
+batch_normalization_1/AssignMovingAvg/sub/xConst*
valueB
 *  ?*>
_class4
20loc:@batch_normalization_1/AssignMovingAvg/24822*
dtype0*
_output_shapes
: 2-
+batch_normalization_1/AssignMovingAvg/sub/x
)batch_normalization_1/AssignMovingAvg/subSub4batch_normalization_1/AssignMovingAvg/sub/x:output:0&batch_normalization_1/Const_2:output:0*
_output_shapes
: *
T0*>
_class4
20loc:@batch_normalization_1/AssignMovingAvg/248222+
)batch_normalization_1/AssignMovingAvg/subд
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp+batch_normalization_1_assignmovingavg_24822*
dtype0*
_output_shapes
: 26
4batch_normalization_1/AssignMovingAvg/ReadVariableOpЙ
+batch_normalization_1/AssignMovingAvg/sub_1Sub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_1/FusedBatchNormV3:batch_mean:0*
T0*>
_class4
20loc:@batch_normalization_1/AssignMovingAvg/24822*
_output_shapes
: 2-
+batch_normalization_1/AssignMovingAvg/sub_1Ђ
)batch_normalization_1/AssignMovingAvg/mulMul/batch_normalization_1/AssignMovingAvg/sub_1:z:0-batch_normalization_1/AssignMovingAvg/sub:z:0*
T0*>
_class4
20loc:@batch_normalization_1/AssignMovingAvg/24822*
_output_shapes
: 2+
)batch_normalization_1/AssignMovingAvg/mul
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp+batch_normalization_1_assignmovingavg_24822-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp*>
_class4
20loc:@batch_normalization_1/AssignMovingAvg/24822*
dtype0*
_output_shapes
 2;
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpх
-batch_normalization_1/AssignMovingAvg_1/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?*@
_class6
42loc:@batch_normalization_1/AssignMovingAvg_1/248292/
-batch_normalization_1/AssignMovingAvg_1/sub/xЄ
+batch_normalization_1/AssignMovingAvg_1/subSub6batch_normalization_1/AssignMovingAvg_1/sub/x:output:0&batch_normalization_1/Const_2:output:0*
T0*@
_class6
42loc:@batch_normalization_1/AssignMovingAvg_1/24829*
_output_shapes
: 2-
+batch_normalization_1/AssignMovingAvg_1/subк
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp-batch_normalization_1_assignmovingavg_1_24829*
dtype0*
_output_shapes
: 28
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpХ
-batch_normalization_1/AssignMovingAvg_1/sub_1Sub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_1/FusedBatchNormV3:batch_variance:0*
_output_shapes
: *
T0*@
_class6
42loc:@batch_normalization_1/AssignMovingAvg_1/248292/
-batch_normalization_1/AssignMovingAvg_1/sub_1Ќ
+batch_normalization_1/AssignMovingAvg_1/mulMul1batch_normalization_1/AssignMovingAvg_1/sub_1:z:0/batch_normalization_1/AssignMovingAvg_1/sub:z:0*
_output_shapes
: *
T0*@
_class6
42loc:@batch_normalization_1/AssignMovingAvg_1/248292-
+batch_normalization_1/AssignMovingAvg_1/mul
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp-batch_normalization_1_assignmovingavg_1_24829/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
dtype0*
_output_shapes
 *@
_class6
42loc:@batch_normalization_1/AssignMovingAvg_1/248292=
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp
activation_2/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
activation_2/Relu
add/addAddV2activation_2/Relu:activations:0max_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2	
add/addu
activation_3/ReluReluadd/add:z:0*/
_output_shapes
:џџџџџџџџџ *
T02
activation_3/ReluА
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*
dtype0*&
_output_shapes
: @2 
conv2d_3/Conv2D/ReadVariableOpз
conv2d_3/Conv2DConv2Dactivation_3/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ@*
T02
conv2d_3/Conv2DЇ
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
dtype0*
_output_shapes
:@2!
conv2d_3/BiasAdd/ReadVariableOpЌ
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:џџџџџџџџџ@*
T02
conv2d_3/BiasAdd
"batch_normalization_3/LogicalAnd/xConst*
dtype0
*
_output_shapes
: *
value	B
 Z2$
"batch_normalization_3/LogicalAnd/x
"batch_normalization_3/LogicalAnd/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z2$
"batch_normalization_3/LogicalAnd/yФ
 batch_normalization_3/LogicalAnd
LogicalAnd+batch_normalization_3/LogicalAnd/x:output:0+batch_normalization_3/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_3/LogicalAndЖ
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
dtype0*
_output_shapes
:@2&
$batch_normalization_3/ReadVariableOpМ
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
dtype0*
_output_shapes
:@2(
&batch_normalization_3/ReadVariableOp_1}
batch_normalization_3/ConstConst*
valueB *
dtype0*
_output_shapes
: 2
batch_normalization_3/Const
batch_normalization_3/Const_1Const*
dtype0*
_output_shapes
: *
valueB 2
batch_normalization_3/Const_1
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_3/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0$batch_normalization_3/Const:output:0&batch_normalization_3/Const_1:output:0*K
_output_shapes9
7:џџџџџџџџџ@:@:@:@:@:*
T0*
U0*
epsilon%o:2(
&batch_normalization_3/FusedBatchNormV3
batch_normalization_3/Const_2Const*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2
batch_normalization_3/Const_2п
+batch_normalization_3/AssignMovingAvg/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?*>
_class4
20loc:@batch_normalization_3/AssignMovingAvg/248612-
+batch_normalization_3/AssignMovingAvg/sub/x
)batch_normalization_3/AssignMovingAvg/subSub4batch_normalization_3/AssignMovingAvg/sub/x:output:0&batch_normalization_3/Const_2:output:0*
T0*>
_class4
20loc:@batch_normalization_3/AssignMovingAvg/24861*
_output_shapes
: 2+
)batch_normalization_3/AssignMovingAvg/subд
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp+batch_normalization_3_assignmovingavg_24861*
dtype0*
_output_shapes
:@26
4batch_normalization_3/AssignMovingAvg/ReadVariableOpЙ
+batch_normalization_3/AssignMovingAvg/sub_1Sub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_3/FusedBatchNormV3:batch_mean:0*
T0*>
_class4
20loc:@batch_normalization_3/AssignMovingAvg/24861*
_output_shapes
:@2-
+batch_normalization_3/AssignMovingAvg/sub_1Ђ
)batch_normalization_3/AssignMovingAvg/mulMul/batch_normalization_3/AssignMovingAvg/sub_1:z:0-batch_normalization_3/AssignMovingAvg/sub:z:0*
_output_shapes
:@*
T0*>
_class4
20loc:@batch_normalization_3/AssignMovingAvg/248612+
)batch_normalization_3/AssignMovingAvg/mul
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp+batch_normalization_3_assignmovingavg_24861-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp*>
_class4
20loc:@batch_normalization_3/AssignMovingAvg/24861*
dtype0*
_output_shapes
 2;
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpх
-batch_normalization_3/AssignMovingAvg_1/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?*@
_class6
42loc:@batch_normalization_3/AssignMovingAvg_1/248682/
-batch_normalization_3/AssignMovingAvg_1/sub/xЄ
+batch_normalization_3/AssignMovingAvg_1/subSub6batch_normalization_3/AssignMovingAvg_1/sub/x:output:0&batch_normalization_3/Const_2:output:0*
T0*@
_class6
42loc:@batch_normalization_3/AssignMovingAvg_1/24868*
_output_shapes
: 2-
+batch_normalization_3/AssignMovingAvg_1/subк
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp-batch_normalization_3_assignmovingavg_1_24868*
dtype0*
_output_shapes
:@28
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpХ
-batch_normalization_3/AssignMovingAvg_1/sub_1Sub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_3/FusedBatchNormV3:batch_variance:0*
T0*@
_class6
42loc:@batch_normalization_3/AssignMovingAvg_1/24868*
_output_shapes
:@2/
-batch_normalization_3/AssignMovingAvg_1/sub_1Ќ
+batch_normalization_3/AssignMovingAvg_1/mulMul1batch_normalization_3/AssignMovingAvg_1/sub_1:z:0/batch_normalization_3/AssignMovingAvg_1/sub:z:0*
_output_shapes
:@*
T0*@
_class6
42loc:@batch_normalization_3/AssignMovingAvg_1/248682-
+batch_normalization_3/AssignMovingAvg_1/mul
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp-batch_normalization_3_assignmovingavg_1_24868/batch_normalization_3/AssignMovingAvg_1/mul:z:07^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp*@
_class6
42loc:@batch_normalization_3/AssignMovingAvg_1/24868*
dtype0*
_output_shapes
 2=
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp
activation_4/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*/
_output_shapes
:џџџџџџџџџ@*
T02
activation_4/ReluА
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*
dtype0*&
_output_shapes
:@@2 
conv2d_4/Conv2D/ReadVariableOpз
conv2d_4/Conv2DConv2Dactivation_4/Relu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ@2
conv2d_4/Conv2DЇ
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
dtype0*
_output_shapes
:@2!
conv2d_4/BiasAdd/ReadVariableOpЌ
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:џџџџџџџџџ@*
T02
conv2d_4/BiasAddА
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*
dtype0*&
_output_shapes
: @2 
conv2d_2/Conv2D/ReadVariableOpз
conv2d_2/Conv2DConv2Dactivation_3/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ@2
conv2d_2/Conv2DЇ
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
dtype0*
_output_shapes
:@2!
conv2d_2/BiasAdd/ReadVariableOpЌ
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:џџџџџџџџџ@*
T02
conv2d_2/BiasAdd
"batch_normalization_4/LogicalAnd/xConst*
dtype0
*
_output_shapes
: *
value	B
 Z2$
"batch_normalization_4/LogicalAnd/x
"batch_normalization_4/LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2$
"batch_normalization_4/LogicalAnd/yФ
 batch_normalization_4/LogicalAnd
LogicalAnd+batch_normalization_4/LogicalAnd/x:output:0+batch_normalization_4/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_4/LogicalAndЖ
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
dtype0*
_output_shapes
:@2&
$batch_normalization_4/ReadVariableOpМ
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
dtype0*
_output_shapes
:@2(
&batch_normalization_4/ReadVariableOp_1}
batch_normalization_4/ConstConst*
dtype0*
_output_shapes
: *
valueB 2
batch_normalization_4/Const
batch_normalization_4/Const_1Const*
valueB *
dtype0*
_output_shapes
: 2
batch_normalization_4/Const_1
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_4/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0$batch_normalization_4/Const:output:0&batch_normalization_4/Const_1:output:0*K
_output_shapes9
7:џџџџџџџџџ@:@:@:@:@:*
T0*
U0*
epsilon%o:2(
&batch_normalization_4/FusedBatchNormV3
batch_normalization_4/Const_2Const*
dtype0*
_output_shapes
: *
valueB
 *Єp}?2
batch_normalization_4/Const_2п
+batch_normalization_4/AssignMovingAvg/sub/xConst*
valueB
 *  ?*>
_class4
20loc:@batch_normalization_4/AssignMovingAvg/24904*
dtype0*
_output_shapes
: 2-
+batch_normalization_4/AssignMovingAvg/sub/x
)batch_normalization_4/AssignMovingAvg/subSub4batch_normalization_4/AssignMovingAvg/sub/x:output:0&batch_normalization_4/Const_2:output:0*
_output_shapes
: *
T0*>
_class4
20loc:@batch_normalization_4/AssignMovingAvg/249042+
)batch_normalization_4/AssignMovingAvg/subд
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp+batch_normalization_4_assignmovingavg_24904*
dtype0*
_output_shapes
:@26
4batch_normalization_4/AssignMovingAvg/ReadVariableOpЙ
+batch_normalization_4/AssignMovingAvg/sub_1Sub<batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_4/FusedBatchNormV3:batch_mean:0*
T0*>
_class4
20loc:@batch_normalization_4/AssignMovingAvg/24904*
_output_shapes
:@2-
+batch_normalization_4/AssignMovingAvg/sub_1Ђ
)batch_normalization_4/AssignMovingAvg/mulMul/batch_normalization_4/AssignMovingAvg/sub_1:z:0-batch_normalization_4/AssignMovingAvg/sub:z:0*
_output_shapes
:@*
T0*>
_class4
20loc:@batch_normalization_4/AssignMovingAvg/249042+
)batch_normalization_4/AssignMovingAvg/mul
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp+batch_normalization_4_assignmovingavg_24904-batch_normalization_4/AssignMovingAvg/mul:z:05^batch_normalization_4/AssignMovingAvg/ReadVariableOp*
dtype0*
_output_shapes
 *>
_class4
20loc:@batch_normalization_4/AssignMovingAvg/249042;
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpх
-batch_normalization_4/AssignMovingAvg_1/sub/xConst*
valueB
 *  ?*@
_class6
42loc:@batch_normalization_4/AssignMovingAvg_1/24911*
dtype0*
_output_shapes
: 2/
-batch_normalization_4/AssignMovingAvg_1/sub/xЄ
+batch_normalization_4/AssignMovingAvg_1/subSub6batch_normalization_4/AssignMovingAvg_1/sub/x:output:0&batch_normalization_4/Const_2:output:0*
_output_shapes
: *
T0*@
_class6
42loc:@batch_normalization_4/AssignMovingAvg_1/249112-
+batch_normalization_4/AssignMovingAvg_1/subк
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp-batch_normalization_4_assignmovingavg_1_24911*
dtype0*
_output_shapes
:@28
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpХ
-batch_normalization_4/AssignMovingAvg_1/sub_1Sub>batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_4/FusedBatchNormV3:batch_variance:0*
T0*@
_class6
42loc:@batch_normalization_4/AssignMovingAvg_1/24911*
_output_shapes
:@2/
-batch_normalization_4/AssignMovingAvg_1/sub_1Ќ
+batch_normalization_4/AssignMovingAvg_1/mulMul1batch_normalization_4/AssignMovingAvg_1/sub_1:z:0/batch_normalization_4/AssignMovingAvg_1/sub:z:0*
_output_shapes
:@*
T0*@
_class6
42loc:@batch_normalization_4/AssignMovingAvg_1/249112-
+batch_normalization_4/AssignMovingAvg_1/mul
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp-batch_normalization_4_assignmovingavg_1_24911/batch_normalization_4/AssignMovingAvg_1/mul:z:07^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*@
_class6
42loc:@batch_normalization_4/AssignMovingAvg_1/24911*
dtype0*
_output_shapes
 2=
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp
activation_5/ReluRelu*batch_normalization_4/FusedBatchNormV3:y:0*/
_output_shapes
:џџџџџџџџџ@*
T02
activation_5/Relu
"batch_normalization_2/LogicalAnd/xConst*
dtype0
*
_output_shapes
: *
value	B
 Z2$
"batch_normalization_2/LogicalAnd/x
"batch_normalization_2/LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2$
"batch_normalization_2/LogicalAnd/yФ
 batch_normalization_2/LogicalAnd
LogicalAnd+batch_normalization_2/LogicalAnd/x:output:0+batch_normalization_2/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_2/LogicalAndЖ
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
dtype0*
_output_shapes
:@2&
$batch_normalization_2/ReadVariableOpМ
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
dtype0*
_output_shapes
:@2(
&batch_normalization_2/ReadVariableOp_1}
batch_normalization_2/ConstConst*
dtype0*
_output_shapes
: *
valueB 2
batch_normalization_2/Const
batch_normalization_2/Const_1Const*
dtype0*
_output_shapes
: *
valueB 2
batch_normalization_2/Const_1
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0$batch_normalization_2/Const:output:0&batch_normalization_2/Const_1:output:0*
epsilon%o:*K
_output_shapes9
7:џџџџџџџџџ@:@:@:@:@:*
T0*
U02(
&batch_normalization_2/FusedBatchNormV3
batch_normalization_2/Const_2Const*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2
batch_normalization_2/Const_2п
+batch_normalization_2/AssignMovingAvg/sub/xConst*
valueB
 *  ?*>
_class4
20loc:@batch_normalization_2/AssignMovingAvg/24935*
dtype0*
_output_shapes
: 2-
+batch_normalization_2/AssignMovingAvg/sub/x
)batch_normalization_2/AssignMovingAvg/subSub4batch_normalization_2/AssignMovingAvg/sub/x:output:0&batch_normalization_2/Const_2:output:0*
T0*>
_class4
20loc:@batch_normalization_2/AssignMovingAvg/24935*
_output_shapes
: 2+
)batch_normalization_2/AssignMovingAvg/subд
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp+batch_normalization_2_assignmovingavg_24935*
dtype0*
_output_shapes
:@26
4batch_normalization_2/AssignMovingAvg/ReadVariableOpЙ
+batch_normalization_2/AssignMovingAvg/sub_1Sub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_2/FusedBatchNormV3:batch_mean:0*
_output_shapes
:@*
T0*>
_class4
20loc:@batch_normalization_2/AssignMovingAvg/249352-
+batch_normalization_2/AssignMovingAvg/sub_1Ђ
)batch_normalization_2/AssignMovingAvg/mulMul/batch_normalization_2/AssignMovingAvg/sub_1:z:0-batch_normalization_2/AssignMovingAvg/sub:z:0*
T0*>
_class4
20loc:@batch_normalization_2/AssignMovingAvg/24935*
_output_shapes
:@2+
)batch_normalization_2/AssignMovingAvg/mul
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp+batch_normalization_2_assignmovingavg_24935-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp*
dtype0*
_output_shapes
 *>
_class4
20loc:@batch_normalization_2/AssignMovingAvg/249352;
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpх
-batch_normalization_2/AssignMovingAvg_1/sub/xConst*
valueB
 *  ?*@
_class6
42loc:@batch_normalization_2/AssignMovingAvg_1/24942*
dtype0*
_output_shapes
: 2/
-batch_normalization_2/AssignMovingAvg_1/sub/xЄ
+batch_normalization_2/AssignMovingAvg_1/subSub6batch_normalization_2/AssignMovingAvg_1/sub/x:output:0&batch_normalization_2/Const_2:output:0*
T0*@
_class6
42loc:@batch_normalization_2/AssignMovingAvg_1/24942*
_output_shapes
: 2-
+batch_normalization_2/AssignMovingAvg_1/subк
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp-batch_normalization_2_assignmovingavg_1_24942*
dtype0*
_output_shapes
:@28
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpХ
-batch_normalization_2/AssignMovingAvg_1/sub_1Sub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_2/FusedBatchNormV3:batch_variance:0*
_output_shapes
:@*
T0*@
_class6
42loc:@batch_normalization_2/AssignMovingAvg_1/249422/
-batch_normalization_2/AssignMovingAvg_1/sub_1Ќ
+batch_normalization_2/AssignMovingAvg_1/mulMul1batch_normalization_2/AssignMovingAvg_1/sub_1:z:0/batch_normalization_2/AssignMovingAvg_1/sub:z:0*
_output_shapes
:@*
T0*@
_class6
42loc:@batch_normalization_2/AssignMovingAvg_1/249422-
+batch_normalization_2/AssignMovingAvg_1/mul
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp-batch_normalization_2_assignmovingavg_1_24942/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *@
_class6
42loc:@batch_normalization_2/AssignMovingAvg_1/24942*
dtype02=
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpІ
	add_1/addAddV2activation_5/Relu:activations:0*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
	add_1/addw
activation_6/ReluReluadd_1/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
activation_6/ReluГ
/global_average_pooling2d/Mean/reduction_indicesConst*
valueB"      *
dtype0*
_output_shapes
:21
/global_average_pooling2d/Mean/reduction_indicesг
global_average_pooling2d/MeanMeanactivation_6/Relu:activations:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
global_average_pooling2d/Meano
flatten/ConstConst*
valueB"џџџџ@   *
dtype0*
_output_shapes
:2
flatten/Const
flatten/ReshapeReshape&global_average_pooling2d/Mean:output:0flatten/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
flatten/Reshape
fc/MatMul/ReadVariableOpReadVariableOp!fc_matmul_readvariableop_resource*
dtype0*
_output_shapes

:@2
fc/MatMul/ReadVariableOp
	fc/MatMulMatMulflatten/Reshape:output:0 fc/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
	fc/MatMul
fc/BiasAdd/ReadVariableOpReadVariableOp"fc_biasadd_readvariableop_resource*
dtype0*
_output_shapes
:2
fc/BiasAdd/ReadVariableOp

fc/BiasAddBiasAddfc/MatMul:product:0!fc/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2

fc/BiasAddj

fc/SigmoidSigmoidfc/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

fc/SigmoidЃ
IdentityIdentityfc/Sigmoid:y:08^batch_normalization/AssignMovingAvg/AssignSubVariableOp3^batch_normalization/AssignMovingAvg/ReadVariableOp:^batch_normalization/AssignMovingAvg_1/AssignSubVariableOp5^batch_normalization/AssignMovingAvg_1/ReadVariableOp#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1:^batch_normalization_1/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_1/AssignMovingAvg/ReadVariableOp<^batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1:^batch_normalization_2/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_2/AssignMovingAvg/ReadVariableOp<^batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1:^batch_normalization_3/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_3/AssignMovingAvg/ReadVariableOp<^batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1:^batch_normalization_4/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_4/AssignMovingAvg/ReadVariableOp<^batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1-^bn_conv1/AssignMovingAvg/AssignSubVariableOp(^bn_conv1/AssignMovingAvg/ReadVariableOp/^bn_conv1/AssignMovingAvg_1/AssignSubVariableOp*^bn_conv1/AssignMovingAvg_1/ReadVariableOp^bn_conv1/ReadVariableOp^bn_conv1/ReadVariableOp_1^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^fc/BiasAdd/ReadVariableOp^fc/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*Ш
_input_shapesЖ
Г:џџџџџџџџџ		::::::::::::::::::::::::::::::::::::::26
bn_conv1/ReadVariableOp_1bn_conv1/ReadVariableOp_12B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp22
bn_conv1/ReadVariableOpbn_conv1/ReadVariableOp2\
,bn_conv1/AssignMovingAvg/AssignSubVariableOp,bn_conv1/AssignMovingAvg/AssignSubVariableOp2v
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_4/AssignMovingAvg/ReadVariableOp4batch_normalization_4/AssignMovingAvg/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12z
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2`
.bn_conv1/AssignMovingAvg_1/AssignSubVariableOp.bn_conv1/AssignMovingAvg_1/AssignSubVariableOp2l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12v
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp2p
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2z
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp24
fc/MatMul/ReadVariableOpfc/MatMul/ReadVariableOp2p
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp2p
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2z
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_1/AssignMovingAvg/ReadVariableOp4batch_normalization_1/AssignMovingAvg/ReadVariableOp2r
7batch_normalization/AssignMovingAvg/AssignSubVariableOp7batch_normalization/AssignMovingAvg/AssignSubVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2z
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp2L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2l
4batch_normalization_3/AssignMovingAvg/ReadVariableOp4batch_normalization_3/AssignMovingAvg/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12V
)bn_conv1/AssignMovingAvg_1/ReadVariableOp)bn_conv1/AssignMovingAvg_1/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp26
fc/BiasAdd/ReadVariableOpfc/BiasAdd/ReadVariableOp2v
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp2R
'bn_conv1/AssignMovingAvg/ReadVariableOp'bn_conv1/AssignMovingAvg/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2l
4batch_normalization_2/AssignMovingAvg/ReadVariableOp4batch_normalization_2/AssignMovingAvg/ReadVariableOp2<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs
Є
c
G__inference_activation_5_layer_call_and_return_conditional_losses_26085

inputs
identityV
ReluReluinputs*/
_output_shapes
:џџџџџџџџџ@*
T02
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:& "
 
_user_specified_nameinputs
К
Ї
&__inference_conv2d_layer_call_fn_22904

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*
Tin
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *,
_gradient_op_typePartitionedCallUnused*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_22896*
Tout
22
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
џ
^
B__inference_flatten_layer_call_and_return_conditional_losses_24278

inputs
identity_
ConstConst*
valueB"џџџџ@   *
dtype0*
_output_shapes
:2
Constg
ReshapeReshapeinputsConst:output:0*'
_output_shapes
:џџџџџџџџџ@*
T02	
Reshaped
IdentityIdentityReshape:output:0*'
_output_shapes
:џџџџџџџџџ@*
T02

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ@:& "
 
_user_specified_nameinputs
ЊІ
Ѓ+
__inference__traced_save_26616
file_prefix+
'savev2_conv1_kernel_read_readvariableop)
%savev2_conv1_bias_read_readvariableop-
)savev2_bn_conv1_gamma_read_readvariableop,
(savev2_bn_conv1_beta_read_readvariableop3
/savev2_bn_conv1_moving_mean_read_readvariableop7
3savev2_bn_conv1_moving_variance_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop:
6savev2_batch_normalization_4_gamma_read_readvariableop9
5savev2_batch_normalization_4_beta_read_readvariableop@
<savev2_batch_normalization_4_moving_mean_read_readvariableopD
@savev2_batch_normalization_4_moving_variance_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop(
$savev2_fc_kernel_read_readvariableop&
"savev2_fc_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop2
.savev2_adam_conv1_kernel_m_read_readvariableop0
,savev2_adam_conv1_bias_m_read_readvariableop4
0savev2_adam_bn_conv1_gamma_m_read_readvariableop3
/savev2_adam_bn_conv1_beta_m_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop?
;savev2_adam_batch_normalization_gamma_m_read_readvariableop>
:savev2_adam_batch_normalization_beta_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_m_read_readvariableop5
1savev2_adam_conv2d_3_kernel_m_read_readvariableop3
/savev2_adam_conv2d_3_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_3_beta_m_read_readvariableop5
1savev2_adam_conv2d_4_kernel_m_read_readvariableop3
/savev2_adam_conv2d_4_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_4_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_4_beta_m_read_readvariableop5
1savev2_adam_conv2d_2_kernel_m_read_readvariableop3
/savev2_adam_conv2d_2_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_m_read_readvariableop/
+savev2_adam_fc_kernel_m_read_readvariableop-
)savev2_adam_fc_bias_m_read_readvariableop2
.savev2_adam_conv1_kernel_v_read_readvariableop0
,savev2_adam_conv1_bias_v_read_readvariableop4
0savev2_adam_bn_conv1_gamma_v_read_readvariableop3
/savev2_adam_bn_conv1_beta_v_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop?
;savev2_adam_batch_normalization_gamma_v_read_readvariableop>
:savev2_adam_batch_normalization_beta_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_v_read_readvariableop5
1savev2_adam_conv2d_3_kernel_v_read_readvariableop3
/savev2_adam_conv2d_3_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_3_beta_v_read_readvariableop5
1savev2_adam_conv2d_4_kernel_v_read_readvariableop3
/savev2_adam_conv2d_4_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_4_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_4_beta_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop3
/savev2_adam_conv2d_2_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_v_read_readvariableop/
+savev2_adam_fc_kernel_v_read_readvariableop-
)savev2_adam_fc_bias_v_read_readvariableop
savev2_1_const

identity_1ЂMergeV2CheckpointsЂSaveV2ЂSaveV2_1Ѕ
StringJoin/inputs_1Const"/device:CPU:0*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_3c5025229ebb49e0af086a33bd83e02a/part2
StringJoin/inputs_1

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 2
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameМ6
SaveV2/tensor_namesConst"/device:CPU:0*Ю5
valueФ5BС5aB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:a2
SaveV2/tensor_namesЭ
SaveV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:a*з
valueЭBЪaB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesЊ)
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_conv1_kernel_read_readvariableop%savev2_conv1_bias_read_readvariableop)savev2_bn_conv1_gamma_read_readvariableop(savev2_bn_conv1_beta_read_readvariableop/savev2_bn_conv1_moving_mean_read_readvariableop3savev2_bn_conv1_moving_variance_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop$savev2_fc_kernel_read_readvariableop"savev2_fc_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop.savev2_adam_conv1_kernel_m_read_readvariableop,savev2_adam_conv1_bias_m_read_readvariableop0savev2_adam_bn_conv1_gamma_m_read_readvariableop/savev2_adam_bn_conv1_beta_m_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop;savev2_adam_batch_normalization_gamma_m_read_readvariableop:savev2_adam_batch_normalization_beta_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop<savev2_adam_batch_normalization_1_beta_m_read_readvariableop1savev2_adam_conv2d_3_kernel_m_read_readvariableop/savev2_adam_conv2d_3_bias_m_read_readvariableop=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop<savev2_adam_batch_normalization_3_beta_m_read_readvariableop1savev2_adam_conv2d_4_kernel_m_read_readvariableop/savev2_adam_conv2d_4_bias_m_read_readvariableop=savev2_adam_batch_normalization_4_gamma_m_read_readvariableop<savev2_adam_batch_normalization_4_beta_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop<savev2_adam_batch_normalization_2_beta_m_read_readvariableop+savev2_adam_fc_kernel_m_read_readvariableop)savev2_adam_fc_bias_m_read_readvariableop.savev2_adam_conv1_kernel_v_read_readvariableop,savev2_adam_conv1_bias_v_read_readvariableop0savev2_adam_bn_conv1_gamma_v_read_readvariableop/savev2_adam_bn_conv1_beta_v_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop;savev2_adam_batch_normalization_gamma_v_read_readvariableop:savev2_adam_batch_normalization_beta_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop<savev2_adam_batch_normalization_1_beta_v_read_readvariableop1savev2_adam_conv2d_3_kernel_v_read_readvariableop/savev2_adam_conv2d_3_bias_v_read_readvariableop=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop<savev2_adam_batch_normalization_3_beta_v_read_readvariableop1savev2_adam_conv2d_4_kernel_v_read_readvariableop/savev2_adam_conv2d_4_bias_v_read_readvariableop=savev2_adam_batch_normalization_4_gamma_v_read_readvariableop<savev2_adam_batch_normalization_4_beta_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableop=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop<savev2_adam_batch_normalization_2_beta_v_read_readvariableop+savev2_adam_fc_kernel_v_read_readvariableop)savev2_adam_fc_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *o
dtypese
c2a	2
SaveV2
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: 2
ShardedFilename_1/shardЌ
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1Ђ
SaveV2_1/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:2
SaveV2_1/shape_and_slicesЯ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1у
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЌ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
_output_shapes
: *
T02

Identity

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Ї
_input_shapes
: : : : : : : :  : : : : : :  : : : : : : @:@:@:@:@:@:@@:@:@:@:@:@: @:@:@:@:@:@:@:: : : : : : : : : : : :  : : : :  : : : : @:@:@:@:@@:@:@:@: @:@:@:@:@:: : : : :  : : : :  : : : : @:@:@:@:@@:@:@:@: @:@:@:@:@:: 2
SaveV2_1SaveV2_12
SaveV2SaveV22(
MergeV2CheckpointsMergeV2Checkpoints:+ '
%
_user_specified_namefile_prefix
ў
ё
N__inference_batch_normalization_layer_call_and_return_conditional_losses_25456

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
dtype0
*
_output_shapes
: *
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
: 2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
: 2
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
dtype0*
_output_shapes
: 2!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
dtype0*
_output_shapes
: 2#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*
is_training( *
epsilon%o:*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :2
FusedBatchNormV3S
ConstConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2
Constь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
T02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::::2 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_1:& "
 
_user_specified_nameinputs
Ї$

N__inference_batch_normalization_layer_call_and_return_conditional_losses_25434

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_25419
assignmovingavg_1_25426
identityЂ#AssignMovingAvg/AssignSubVariableOpЂAssignMovingAvg/ReadVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpЂ AssignMovingAvg_1/ReadVariableOpЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
dtype0
*
_output_shapes
: *
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
dtype0*
_output_shapes
: 2
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Q
ConstConst*
valueB *
dtype0*
_output_shapes
: 2
ConstU
Const_1Const*
valueB *
dtype0*
_output_shapes
: 2	
Const_1
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
T0*
U0*
epsilon%o:2
FusedBatchNormV3W
Const_2Const*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2	
Const_2
AssignMovingAvg/sub/xConst*
valueB
 *  ?*(
_class
loc:@AssignMovingAvg/25419*
dtype0*
_output_shapes
: 2
AssignMovingAvg/sub/xЎ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
_output_shapes
: *
T0*(
_class
loc:@AssignMovingAvg/254192
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_25419*
dtype0*
_output_shapes
: 2 
AssignMovingAvg/ReadVariableOpЫ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/25419*
_output_shapes
: 2
AssignMovingAvg/sub_1Д
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/25419*
_output_shapes
: 2
AssignMovingAvg/mulџ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_25419AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
dtype0*
_output_shapes
 *(
_class
loc:@AssignMovingAvg/254192%
#AssignMovingAvg/AssignSubVariableOpЃ
AssignMovingAvg_1/sub/xConst*
valueB
 *  ?**
_class 
loc:@AssignMovingAvg_1/25426*
dtype0*
_output_shapes
: 2
AssignMovingAvg_1/sub/xЖ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/25426*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_25426*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpз
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/25426*
_output_shapes
: 2
AssignMovingAvg_1/sub_1О
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/25426*
_output_shapes
: 2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_25426AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/25426*
dtype0*
_output_shapes
 2'
%AssignMovingAvg_1/AssignSubVariableOpИ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
T02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::::2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
ѓ
ц
C__inference_bn_conv1_layer_call_and_return_conditional_losses_22866

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: 2
LogicalAnd/x^
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
dtype0*
_output_shapes
: 2
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
dtype0*
_output_shapes
: 2!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
dtype0*
_output_shapes
: 2#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
T0*
U0*
is_training( *
epsilon%o:2
FusedBatchNormV3S
ConstConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 2
Constь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::::2 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_1:& "
 
_user_specified_nameinputs
ъ
ў
5__inference_batch_normalization_3_layer_call_fn_25827

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityЂStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin	
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*,
_gradient_op_typePartitionedCallUnused*Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_233002
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs"ЏL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*­
serving_default
C
input_18
serving_default_input_1:0џџџџџџџџџ		6
fc0
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:Ѓ
ЪД
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer-11
layer-12
layer_with_weights-6
layer-13
layer_with_weights-7
layer-14
layer-15
layer_with_weights-8
layer-16
layer_with_weights-9
layer-17
layer_with_weights-10
layer-18
layer-19
layer_with_weights-11
layer-20
layer-21
layer-22
layer-23
layer-24
layer_with_weights-12
layer-25
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api
 
signatures
о__call__
+п&call_and_return_all_conditional_losses
р_default_save_signature"Ф­
_tf_keras_modelЉ­{"class_name": "Model", "name": "ResNet9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "ResNet9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 9, 9, 7], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [5, 5], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "bn_conv1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "bn_conv1", "inbound_nodes": [[["conv1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["bn_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [3, 3], "padding": "same", "strides": [2, 2], "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["max_pooling2d", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [2, 2], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_2", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add", "trainable": true, "dtype": "float32"}, "name": "add", "inbound_nodes": [[["activation_2", 0, 0, {}], ["max_pooling2d", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_3", "inbound_nodes": [[["add", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["activation_3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_3", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_4", "inbound_nodes": [[["batch_normalization_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [2, 2], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["activation_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_4", "inbound_nodes": [[["conv2d_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [1, 1], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["activation_3", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_5", "inbound_nodes": [[["batch_normalization_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "name": "add_1", "inbound_nodes": [[["activation_5", 0, 0, {}], ["batch_normalization_2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_6", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_6", "inbound_nodes": [[["add_1", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling2d", "inbound_nodes": [[["activation_6", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["global_average_pooling2d", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc", "trainable": true, "dtype": "float32", "units": 2, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc", "inbound_nodes": [[["flatten", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["fc", 0, 0]]}, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "ResNet9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 9, 9, 7], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [5, 5], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "bn_conv1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "bn_conv1", "inbound_nodes": [[["conv1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["bn_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [3, 3], "padding": "same", "strides": [2, 2], "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["max_pooling2d", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [2, 2], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_2", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add", "trainable": true, "dtype": "float32"}, "name": "add", "inbound_nodes": [[["activation_2", 0, 0, {}], ["max_pooling2d", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_3", "inbound_nodes": [[["add", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["activation_3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_3", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_4", "inbound_nodes": [[["batch_normalization_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [2, 2], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["activation_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_4", "inbound_nodes": [[["conv2d_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [1, 1], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["activation_3", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_5", "inbound_nodes": [[["batch_normalization_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "name": "add_1", "inbound_nodes": [[["activation_5", 0, 0, {}], ["batch_normalization_2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_6", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_6", "inbound_nodes": [[["add_1", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling2d", "inbound_nodes": [[["activation_6", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["global_average_pooling2d", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc", "trainable": true, "dtype": "float32", "units": 2, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc", "inbound_nodes": [[["flatten", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["fc", 0, 0]]}}, "training_config": {"loss": "binary_crossentropy", "metrics": ["binary_accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
Љ"І
_tf_keras_input_layer{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 9, 9, 7], "config": {"batch_input_shape": [null, 9, 9, 7], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
щ

!kernel
"bias
#trainable_variables
$	variables
%regularization_losses
&	keras_api
с__call__
+т&call_and_return_all_conditional_losses"Т
_tf_keras_layerЈ{"class_name": "Conv2D", "name": "conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [5, 5], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 7}}}}

'axis
	(gamma
)beta
*moving_mean
+moving_variance
,trainable_variables
-	variables
.regularization_losses
/	keras_api
у__call__
+ф&call_and_return_all_conditional_losses"Х
_tf_keras_layerЋ{"class_name": "BatchNormalization", "name": "bn_conv1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "bn_conv1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}}

0trainable_variables
1	variables
2regularization_losses
3	keras_api
х__call__
+ц&call_and_return_all_conditional_losses"
_tf_keras_layerђ{"class_name": "Activation", "name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}}
њ
4trainable_variables
5	variables
6regularization_losses
7	keras_api
ч__call__
+ш&call_and_return_all_conditional_losses"щ
_tf_keras_layerЯ{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [3, 3], "padding": "same", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ь

8kernel
9bias
:trainable_variables
;	variables
<regularization_losses
=	keras_api
щ__call__
+ъ&call_and_return_all_conditional_losses"Х
_tf_keras_layerЋ{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
Б
>axis
	?gamma
@beta
Amoving_mean
Bmoving_variance
Ctrainable_variables
D	variables
Eregularization_losses
F	keras_api
ы__call__
+ь&call_and_return_all_conditional_losses"л
_tf_keras_layerС{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}}
Ё
Gtrainable_variables
H	variables
Iregularization_losses
J	keras_api
э__call__
+ю&call_and_return_all_conditional_losses"
_tf_keras_layerі{"class_name": "Activation", "name": "activation_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}}
№

Kkernel
Lbias
Mtrainable_variables
N	variables
Oregularization_losses
P	keras_api
я__call__
+№&call_and_return_all_conditional_losses"Щ
_tf_keras_layerЏ{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [2, 2], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
Е
Qaxis
	Rgamma
Sbeta
Tmoving_mean
Umoving_variance
Vtrainable_variables
W	variables
Xregularization_losses
Y	keras_api
ё__call__
+ђ&call_and_return_all_conditional_losses"п
_tf_keras_layerХ{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}}
Ё
Ztrainable_variables
[	variables
\regularization_losses
]	keras_api
ѓ__call__
+є&call_and_return_all_conditional_losses"
_tf_keras_layerі{"class_name": "Activation", "name": "activation_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}}
ђ
^trainable_variables
_	variables
`regularization_losses
a	keras_api
ѕ__call__
+і&call_and_return_all_conditional_losses"с
_tf_keras_layerЧ{"class_name": "Add", "name": "add", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "add", "trainable": true, "dtype": "float32"}}
Ё
btrainable_variables
c	variables
dregularization_losses
e	keras_api
ї__call__
+ј&call_and_return_all_conditional_losses"
_tf_keras_layerі{"class_name": "Activation", "name": "activation_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}}
№

fkernel
gbias
htrainable_variables
i	variables
jregularization_losses
k	keras_api
љ__call__
+њ&call_and_return_all_conditional_losses"Щ
_tf_keras_layerЏ{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
Е
laxis
	mgamma
nbeta
omoving_mean
pmoving_variance
qtrainable_variables
r	variables
sregularization_losses
t	keras_api
ћ__call__
+ќ&call_and_return_all_conditional_losses"п
_tf_keras_layerХ{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}}
Ё
utrainable_variables
v	variables
wregularization_losses
x	keras_api
§__call__
+ў&call_and_return_all_conditional_losses"
_tf_keras_layerі{"class_name": "Activation", "name": "activation_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}}
№

ykernel
zbias
{trainable_variables
|	variables
}regularization_losses
~	keras_api
џ__call__
+&call_and_return_all_conditional_losses"Щ
_tf_keras_layerЏ{"class_name": "Conv2D", "name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [2, 2], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
Н
axis

gamma
	beta
moving_mean
moving_variance
trainable_variables
	variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"п
_tf_keras_layerХ{"class_name": "BatchNormalization", "name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}}
і
kernel
	bias
trainable_variables
	variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"Щ
_tf_keras_layerЏ{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [1, 1], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
Ѕ
trainable_variables
	variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layerі{"class_name": "Activation", "name": "activation_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "relu"}}
О
	axis

gamma
	beta
moving_mean
moving_variance
trainable_variables
	variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"п
_tf_keras_layerХ{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}}
њ
trainable_variables
	variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"х
_tf_keras_layerЫ{"class_name": "Add", "name": "add_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "add_1", "trainable": true, "dtype": "float32"}}
Ѕ
trainable_variables
 	variables
Ёregularization_losses
Ђ	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layerі{"class_name": "Activation", "name": "activation_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation_6", "trainable": true, "dtype": "float32", "activation": "relu"}}
у
Ѓtrainable_variables
Є	variables
Ѕregularization_losses
І	keras_api
__call__
+&call_and_return_all_conditional_losses"Ю
_tf_keras_layerД{"class_name": "GlobalAveragePooling2D", "name": "global_average_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
В
Їtrainable_variables
Ј	variables
Љregularization_losses
Њ	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ё
Ћkernel
	Ќbias
­trainable_variables
Ў	variables
Џregularization_losses
А	keras_api
__call__
+&call_and_return_all_conditional_losses"Ф
_tf_keras_layerЊ{"class_name": "Dense", "name": "fc", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "fc", "trainable": true, "dtype": "float32", "units": 2, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}}
№
	Бiter
Вbeta_1
Гbeta_2

Дdecay
Еlearning_rate!mЊ"mЋ(mЌ)m­8mЎ9mЏ?mА@mБKmВLmГRmДSmЕfmЖgmЗmmИnmЙymКzmЛ	mМ	mН	mО	mП	mР	mС	ЋmТ	ЌmУ!vФ"vХ(vЦ)vЧ8vШ9vЩ?vЪ@vЫKvЬLvЭRvЮSvЯfvаgvбmvвnvгyvдzvе	vж	vз	vи	vй	vк	vл	Ћvм	Ќvн"
	optimizer
ю
!0
"1
(2
)3
84
95
?6
@7
K8
L9
R10
S11
f12
g13
m14
n15
y16
z17
18
19
20
21
22
23
Ћ24
Ќ25"
trackable_list_wrapper
в
!0
"1
(2
)3
*4
+5
86
97
?8
@9
A10
B11
K12
L13
R14
S15
T16
U17
f18
g19
m20
n21
o22
p23
y24
z25
26
27
28
29
30
31
32
33
34
35
Ћ36
Ќ37"
trackable_list_wrapper
 "
trackable_list_wrapper
П
Жmetrics
trainable_variables
Зlayers
 Иlayer_regularization_losses
	variables
regularization_losses
Йnon_trainable_variables
о__call__
р_default_save_signature
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
&:$ 2conv1/kernel
: 2
conv1/bias
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ё
Кmetrics
#trainable_variables
Лlayers
 Мlayer_regularization_losses
$	variables
%regularization_losses
Нnon_trainable_variables
с__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
: 2bn_conv1/gamma
: 2bn_conv1/beta
$:"  (2bn_conv1/moving_mean
(:&  (2bn_conv1/moving_variance
.
(0
)1"
trackable_list_wrapper
<
(0
)1
*2
+3"
trackable_list_wrapper
 "
trackable_list_wrapper
Ё
Оmetrics
,trainable_variables
Пlayers
 Рlayer_regularization_losses
-	variables
.regularization_losses
Сnon_trainable_variables
у__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ё
Тmetrics
0trainable_variables
Уlayers
 Фlayer_regularization_losses
1	variables
2regularization_losses
Хnon_trainable_variables
х__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ё
Цmetrics
4trainable_variables
Чlayers
 Шlayer_regularization_losses
5	variables
6regularization_losses
Щnon_trainable_variables
ч__call__
+ш&call_and_return_all_conditional_losses
'ш"call_and_return_conditional_losses"
_generic_user_object
':%  2conv2d/kernel
: 2conv2d/bias
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
Ё
Ъmetrics
:trainable_variables
Ыlayers
 Ьlayer_regularization_losses
;	variables
<regularization_losses
Эnon_trainable_variables
щ__call__
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':% 2batch_normalization/gamma
&:$ 2batch_normalization/beta
/:-  (2batch_normalization/moving_mean
3:1  (2#batch_normalization/moving_variance
.
?0
@1"
trackable_list_wrapper
<
?0
@1
A2
B3"
trackable_list_wrapper
 "
trackable_list_wrapper
Ё
Юmetrics
Ctrainable_variables
Яlayers
 аlayer_regularization_losses
D	variables
Eregularization_losses
бnon_trainable_variables
ы__call__
+ь&call_and_return_all_conditional_losses
'ь"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ё
вmetrics
Gtrainable_variables
гlayers
 дlayer_regularization_losses
H	variables
Iregularization_losses
еnon_trainable_variables
э__call__
+ю&call_and_return_all_conditional_losses
'ю"call_and_return_conditional_losses"
_generic_user_object
):'  2conv2d_1/kernel
: 2conv2d_1/bias
.
K0
L1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ё
жmetrics
Mtrainable_variables
зlayers
 иlayer_regularization_losses
N	variables
Oregularization_losses
йnon_trainable_variables
я__call__
+№&call_and_return_all_conditional_losses
'№"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_1/gamma
(:& 2batch_normalization_1/beta
1:/  (2!batch_normalization_1/moving_mean
5:3  (2%batch_normalization_1/moving_variance
.
R0
S1"
trackable_list_wrapper
<
R0
S1
T2
U3"
trackable_list_wrapper
 "
trackable_list_wrapper
Ё
кmetrics
Vtrainable_variables
лlayers
 мlayer_regularization_losses
W	variables
Xregularization_losses
нnon_trainable_variables
ё__call__
+ђ&call_and_return_all_conditional_losses
'ђ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ё
оmetrics
Ztrainable_variables
пlayers
 рlayer_regularization_losses
[	variables
\regularization_losses
сnon_trainable_variables
ѓ__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ё
тmetrics
^trainable_variables
уlayers
 фlayer_regularization_losses
_	variables
`regularization_losses
хnon_trainable_variables
ѕ__call__
+і&call_and_return_all_conditional_losses
'і"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ё
цmetrics
btrainable_variables
чlayers
 шlayer_regularization_losses
c	variables
dregularization_losses
щnon_trainable_variables
ї__call__
+ј&call_and_return_all_conditional_losses
'ј"call_and_return_conditional_losses"
_generic_user_object
):' @2conv2d_3/kernel
:@2conv2d_3/bias
.
f0
g1"
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ё
ъmetrics
htrainable_variables
ыlayers
 ьlayer_regularization_losses
i	variables
jregularization_losses
эnon_trainable_variables
љ__call__
+њ&call_and_return_all_conditional_losses
'њ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_3/gamma
(:&@2batch_normalization_3/beta
1:/@ (2!batch_normalization_3/moving_mean
5:3@ (2%batch_normalization_3/moving_variance
.
m0
n1"
trackable_list_wrapper
<
m0
n1
o2
p3"
trackable_list_wrapper
 "
trackable_list_wrapper
Ё
юmetrics
qtrainable_variables
яlayers
 №layer_regularization_losses
r	variables
sregularization_losses
ёnon_trainable_variables
ћ__call__
+ќ&call_and_return_all_conditional_losses
'ќ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ё
ђmetrics
utrainable_variables
ѓlayers
 єlayer_regularization_losses
v	variables
wregularization_losses
ѕnon_trainable_variables
§__call__
+ў&call_and_return_all_conditional_losses
'ў"call_and_return_conditional_losses"
_generic_user_object
):'@@2conv2d_4/kernel
:@2conv2d_4/bias
.
y0
z1"
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ё
іmetrics
{trainable_variables
їlayers
 јlayer_regularization_losses
|	variables
}regularization_losses
љnon_trainable_variables
џ__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_4/gamma
(:&@2batch_normalization_4/beta
1:/@ (2!batch_normalization_4/moving_mean
5:3@ (2%batch_normalization_4/moving_variance
0
0
1"
trackable_list_wrapper
@
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
њmetrics
trainable_variables
ћlayers
 ќlayer_regularization_losses
	variables
regularization_losses
§non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
):' @2conv2d_2/kernel
:@2conv2d_2/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
ўmetrics
trainable_variables
џlayers
 layer_regularization_losses
	variables
regularization_losses
non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
metrics
trainable_variables
layers
 layer_regularization_losses
	variables
regularization_losses
non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_2/gamma
(:&@2batch_normalization_2/beta
1:/@ (2!batch_normalization_2/moving_mean
5:3@ (2%batch_normalization_2/moving_variance
0
0
1"
trackable_list_wrapper
@
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
metrics
trainable_variables
layers
 layer_regularization_losses
	variables
regularization_losses
non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
metrics
trainable_variables
layers
 layer_regularization_losses
	variables
regularization_losses
non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
metrics
trainable_variables
layers
 layer_regularization_losses
 	variables
Ёregularization_losses
non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
metrics
Ѓtrainable_variables
layers
 layer_regularization_losses
Є	variables
Ѕregularization_losses
non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
metrics
Їtrainable_variables
layers
 layer_regularization_losses
Ј	variables
Љregularization_losses
non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:@2	fc/kernel
:2fc/bias
0
Ћ0
Ќ1"
trackable_list_wrapper
0
Ћ0
Ќ1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
metrics
­trainable_variables
layers
 layer_regularization_losses
Ў	variables
Џregularization_losses
non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
(
0"
trackable_list_wrapper
ц
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25"
trackable_list_wrapper
 "
trackable_list_wrapper
z
*0
+1
A2
B3
T4
U5
o6
p7
8
9
10
11"
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
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
*0
+1"
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
trackable_list_wrapper
.
A0
B1"
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
trackable_list_wrapper
 "
trackable_list_wrapper
.
T0
U1"
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
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
o0
p1"
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
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
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
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
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
trackable_list_wrapper
Б

total

 count
Ё
_fn_kwargs
Ђtrainable_variables
Ѓ	variables
Єregularization_losses
Ѕ	keras_api
__call__
+&call_and_return_all_conditional_losses"ѓ
_tf_keras_layerй{"class_name": "MeanMetricWrapper", "name": "binary_accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "binary_accuracy", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Іmetrics
Ђtrainable_variables
Їlayers
 Јlayer_regularization_losses
Ѓ	variables
Єregularization_losses
Љnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
 1"
trackable_list_wrapper
+:) 2Adam/conv1/kernel/m
: 2Adam/conv1/bias/m
!: 2Adam/bn_conv1/gamma/m
 : 2Adam/bn_conv1/beta/m
,:*  2Adam/conv2d/kernel/m
: 2Adam/conv2d/bias/m
,:* 2 Adam/batch_normalization/gamma/m
+:) 2Adam/batch_normalization/beta/m
.:,  2Adam/conv2d_1/kernel/m
 : 2Adam/conv2d_1/bias/m
.:, 2"Adam/batch_normalization_1/gamma/m
-:+ 2!Adam/batch_normalization_1/beta/m
.:, @2Adam/conv2d_3/kernel/m
 :@2Adam/conv2d_3/bias/m
.:,@2"Adam/batch_normalization_3/gamma/m
-:+@2!Adam/batch_normalization_3/beta/m
.:,@@2Adam/conv2d_4/kernel/m
 :@2Adam/conv2d_4/bias/m
.:,@2"Adam/batch_normalization_4/gamma/m
-:+@2!Adam/batch_normalization_4/beta/m
.:, @2Adam/conv2d_2/kernel/m
 :@2Adam/conv2d_2/bias/m
.:,@2"Adam/batch_normalization_2/gamma/m
-:+@2!Adam/batch_normalization_2/beta/m
 :@2Adam/fc/kernel/m
:2Adam/fc/bias/m
+:) 2Adam/conv1/kernel/v
: 2Adam/conv1/bias/v
!: 2Adam/bn_conv1/gamma/v
 : 2Adam/bn_conv1/beta/v
,:*  2Adam/conv2d/kernel/v
: 2Adam/conv2d/bias/v
,:* 2 Adam/batch_normalization/gamma/v
+:) 2Adam/batch_normalization/beta/v
.:,  2Adam/conv2d_1/kernel/v
 : 2Adam/conv2d_1/bias/v
.:, 2"Adam/batch_normalization_1/gamma/v
-:+ 2!Adam/batch_normalization_1/beta/v
.:, @2Adam/conv2d_3/kernel/v
 :@2Adam/conv2d_3/bias/v
.:,@2"Adam/batch_normalization_3/gamma/v
-:+@2!Adam/batch_normalization_3/beta/v
.:,@@2Adam/conv2d_4/kernel/v
 :@2Adam/conv2d_4/bias/v
.:,@2"Adam/batch_normalization_4/gamma/v
-:+@2!Adam/batch_normalization_4/beta/v
.:, @2Adam/conv2d_2/kernel/v
 :@2Adam/conv2d_2/bias/v
.:,@2"Adam/batch_normalization_2/gamma/v
-:+@2!Adam/batch_normalization_2/beta/v
 :@2Adam/fc/kernel/v
:2Adam/fc/bias/v
ъ2ч
'__inference_ResNet9_layer_call_fn_24598
'__inference_ResNet9_layer_call_fn_25175
'__inference_ResNet9_layer_call_fn_24488
'__inference_ResNet9_layer_call_fn_25218Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ж2г
B__inference_ResNet9_layer_call_and_return_conditional_losses_24377
B__inference_ResNet9_layer_call_and_return_conditional_losses_24310
B__inference_ResNet9_layer_call_and_return_conditional_losses_25132
B__inference_ResNet9_layer_call_and_return_conditional_losses_24963Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ц2у
 __inference__wrapped_model_22722О
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *.Ђ+
)&
input_1џџџџџџџџџ		
2
%__inference_conv1_layer_call_fn_22741з
В
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
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
@__inference_conv1_layer_call_and_return_conditional_losses_22733з
В
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
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
т2п
(__inference_bn_conv1_layer_call_fn_25295
(__inference_bn_conv1_layer_call_fn_25378
(__inference_bn_conv1_layer_call_fn_25369
(__inference_bn_conv1_layer_call_fn_25304Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ю2Ы
C__inference_bn_conv1_layer_call_and_return_conditional_losses_25360
C__inference_bn_conv1_layer_call_and_return_conditional_losses_25264
C__inference_bn_conv1_layer_call_and_return_conditional_losses_25286
C__inference_bn_conv1_layer_call_and_return_conditional_losses_25338Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
д2б
*__inference_activation_layer_call_fn_25388Ђ
В
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
annotationsЊ *
 
я2ь
E__inference_activation_layer_call_and_return_conditional_losses_25383Ђ
В
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
annotationsЊ *
 
2
-__inference_max_pooling2d_layer_call_fn_22885р
В
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
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
А2­
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_22879р
В
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
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
&__inference_conv2d_layer_call_fn_22904з
В
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
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 2
A__inference_conv2d_layer_call_and_return_conditional_losses_22896з
В
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
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
2
3__inference_batch_normalization_layer_call_fn_25465
3__inference_batch_normalization_layer_call_fn_25539
3__inference_batch_normalization_layer_call_fn_25474
3__inference_batch_normalization_layer_call_fn_25548Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
њ2ї
N__inference_batch_normalization_layer_call_and_return_conditional_losses_25434
N__inference_batch_normalization_layer_call_and_return_conditional_losses_25530
N__inference_batch_normalization_layer_call_and_return_conditional_losses_25508
N__inference_batch_normalization_layer_call_and_return_conditional_losses_25456Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ж2г
,__inference_activation_1_layer_call_fn_25558Ђ
В
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
annotationsЊ *
 
ё2ю
G__inference_activation_1_layer_call_and_return_conditional_losses_25553Ђ
В
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
annotationsЊ *
 
2
(__inference_conv2d_1_layer_call_fn_23055з
В
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
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Ђ2
C__inference_conv2d_1_layer_call_and_return_conditional_losses_23047з
В
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
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
2
5__inference_batch_normalization_1_layer_call_fn_25709
5__inference_batch_normalization_1_layer_call_fn_25635
5__inference_batch_normalization_1_layer_call_fn_25718
5__inference_batch_normalization_1_layer_call_fn_25644Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2џ
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_25678
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_25626
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_25700
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_25604Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ж2г
,__inference_activation_2_layer_call_fn_25728Ђ
В
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
annotationsЊ *
 
ё2ю
G__inference_activation_2_layer_call_and_return_conditional_losses_25723Ђ
В
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
annotationsЊ *
 
Э2Ъ
#__inference_add_layer_call_fn_25740Ђ
В
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
annotationsЊ *
 
ш2х
>__inference_add_layer_call_and_return_conditional_losses_25734Ђ
В
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
annotationsЊ *
 
ж2г
,__inference_activation_3_layer_call_fn_25750Ђ
В
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
annotationsЊ *
 
ё2ю
G__inference_activation_3_layer_call_and_return_conditional_losses_25745Ђ
В
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
annotationsЊ *
 
2
(__inference_conv2d_3_layer_call_fn_23206з
В
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
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Ђ2
C__inference_conv2d_3_layer_call_and_return_conditional_losses_23198з
В
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
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
2
5__inference_batch_normalization_3_layer_call_fn_25910
5__inference_batch_normalization_3_layer_call_fn_25901
5__inference_batch_normalization_3_layer_call_fn_25836
5__inference_batch_normalization_3_layer_call_fn_25827Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2џ
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_25818
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_25892
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_25796
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_25870Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ж2г
,__inference_activation_4_layer_call_fn_25920Ђ
В
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
annotationsЊ *
 
ё2ю
G__inference_activation_4_layer_call_and_return_conditional_losses_25915Ђ
В
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
annotationsЊ *
 
2
(__inference_conv2d_4_layer_call_fn_23357з
В
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
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Ђ2
C__inference_conv2d_4_layer_call_and_return_conditional_losses_23349з
В
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
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
2
5__inference_batch_normalization_4_layer_call_fn_26071
5__inference_batch_normalization_4_layer_call_fn_25997
5__inference_batch_normalization_4_layer_call_fn_26006
5__inference_batch_normalization_4_layer_call_fn_26080Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2џ
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_25988
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_25966
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_26040
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_26062Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
(__inference_conv2d_2_layer_call_fn_23508з
В
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
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Ђ2
C__inference_conv2d_2_layer_call_and_return_conditional_losses_23500з
В
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
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
ж2г
,__inference_activation_5_layer_call_fn_26090Ђ
В
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
annotationsЊ *
 
ё2ю
G__inference_activation_5_layer_call_and_return_conditional_losses_26085Ђ
В
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
annotationsЊ *
 
2
5__inference_batch_normalization_2_layer_call_fn_26250
5__inference_batch_normalization_2_layer_call_fn_26176
5__inference_batch_normalization_2_layer_call_fn_26167
5__inference_batch_normalization_2_layer_call_fn_26241Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2џ
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_26136
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_26158
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_26210
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_26232Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Я2Ь
%__inference_add_1_layer_call_fn_26262Ђ
В
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
annotationsЊ *
 
ъ2ч
@__inference_add_1_layer_call_and_return_conditional_losses_26256Ђ
В
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
annotationsЊ *
 
ж2г
,__inference_activation_6_layer_call_fn_26272Ђ
В
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
annotationsЊ *
 
ё2ю
G__inference_activation_6_layer_call_and_return_conditional_losses_26267Ђ
В
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
annotationsЊ *
 
 2
8__inference_global_average_pooling2d_layer_call_fn_23653р
В
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
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Л2И
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_23647р
В
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
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
б2Ю
'__inference_flatten_layer_call_fn_26283Ђ
В
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
annotationsЊ *
 
ь2щ
B__inference_flatten_layer_call_and_return_conditional_losses_26278Ђ
В
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
annotationsЊ *
 
Ь2Щ
"__inference_fc_layer_call_fn_26301Ђ
В
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
annotationsЊ *
 
ч2ф
=__inference_fc_layer_call_and_return_conditional_losses_26294Ђ
В
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
annotationsЊ *
 
2B0
#__inference_signature_wrapper_24722input_1
Ь2ЩЦ
НВЙ
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
Ь2ЩЦ
НВЙ
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 я
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_26210MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 ы
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_25678RSTUMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 Ъ
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_26158v;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@
p 
Њ "-Ђ*
# 
0џџџџџџџџџ@
 Ъ
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_26062v;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@
p 
Њ "-Ђ*
# 
0џџџџџџџџџ@
 Ч
5__inference_batch_normalization_4_layer_call_fn_25997MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
5__inference_batch_normalization_1_layer_call_fn_25635eRSTU;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ 
p
Њ " џџџџџџџџџ 
,__inference_activation_6_layer_call_fn_26272[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ " џџџџџџџџџ@
5__inference_batch_normalization_1_layer_call_fn_25644eRSTU;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ 
p 
Њ " џџџџџџџџџ я
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_26232MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 У
5__inference_batch_normalization_1_layer_call_fn_25709RSTUMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ С
3__inference_batch_normalization_layer_call_fn_25465?@ABMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ У
5__inference_batch_normalization_1_layer_call_fn_25718RSTUMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ р
@__inference_add_1_layer_call_and_return_conditional_losses_26256jЂg
`Ђ]
[X
*'
inputs/0џџџџџџџџџ@
*'
inputs/1џџџџџџџџџ@
Њ "-Ђ*
# 
0џџџџџџџџџ@
 С
3__inference_batch_normalization_layer_call_fn_25474?@ABMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ Ъ
#__inference_signature_wrapper_24722Ђ2!"()*+89?@ABKLRSTUfgmnopyzЋЌCЂ@
Ђ 
9Њ6
4
input_1)&
input_1џџџџџџџџџ		"'Њ$
"
fc
fcџџџџџџџџџw
"__inference_fc_layer_call_fn_26301QЋЌ/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "џџџџџџџџџ
3__inference_batch_normalization_layer_call_fn_25539e?@AB;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ 
p
Њ " џџџџџџџџџ ­
%__inference_conv1_layer_call_fn_22741!"IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ у
B__inference_ResNet9_layer_call_and_return_conditional_losses_251322!"()*+89?@ABKLRSTUfgmnopyzЋЌ?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ		
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 
3__inference_batch_normalization_layer_call_fn_25548e?@AB;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ 
p 
Њ " џџџџџџџџџ ы
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_25818mnopMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 Ж
#__inference_add_layer_call_fn_25740jЂg
`Ђ]
[X
*'
inputs/0џџџџџџџџџ 
*'
inputs/1џџџџџџџџџ 
Њ " џџџџџџџџџ А
(__inference_conv2d_4_layer_call_fn_23357yzIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
,__inference_activation_2_layer_call_fn_25728[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ " џџџџџџџџџ ы
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_25796mnopMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 и
C__inference_conv2d_3_layer_call_and_return_conditional_losses_23198fgIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 М
 __inference__wrapped_model_227222!"()*+89?@ABKLRSTUfgmnopyzЋЌ8Ђ5
.Ђ+
)&
input_1џџџџџџџџџ		
Њ "'Њ$
"
fc
fcџџџџџџџџџ
*__inference_activation_layer_call_fn_25388[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ " џџџџџџџџџ Б
E__inference_activation_layer_call_and_return_conditional_losses_25383h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ "-Ђ*
# 
0џџџџџџџџџ 
 Ц
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_25870rmnop;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@
p
Њ "-Ђ*
# 
0џџџџџџџџџ@
 v
'__inference_flatten_layer_call_fn_26283K/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "џџџџџџџџџ@
=__inference_fc_layer_call_and_return_conditional_losses_26294^ЋЌ/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "%Ђ"

0џџџџџџџџџ
 Ђ
5__inference_batch_normalization_2_layer_call_fn_26167i;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@
p
Њ " џџџџџџџџџ@Ђ
5__inference_batch_normalization_2_layer_call_fn_26176i;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@
p 
Њ " џџџџџџџџџ@Ц
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_25892rmnop;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@
p 
Њ "-Ђ*
# 
0џџџџџџџџџ@
 Ў
&__inference_conv2d_layer_call_fn_2290489IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ А
(__inference_conv2d_1_layer_call_fn_23055KLIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ Ч
5__inference_batch_normalization_2_layer_call_fn_26241MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@Ч
5__inference_batch_normalization_2_layer_call_fn_26250MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@И
%__inference_add_1_layer_call_fn_26262jЂg
`Ђ]
[X
*'
inputs/0џџџџџџџџџ@
*'
inputs/1џџџџџџџџџ@
Њ " џџџџџџџџџ@к
C__inference_conv2d_2_layer_call_and_return_conditional_losses_23500IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 Ж
(__inference_bn_conv1_layer_call_fn_25304()*+MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ и
C__inference_conv2d_4_layer_call_and_return_conditional_losses_23349yzIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
,__inference_activation_3_layer_call_fn_25750[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ " џџџџџџџџџ ы
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_22879RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 я
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_25966MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 е
@__inference_conv1_layer_call_and_return_conditional_losses_22733!"IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 Г
G__inference_activation_1_layer_call_and_return_conditional_losses_25553h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ "-Ђ*
# 
0џџџџџџџџџ 
 Ж
(__inference_bn_conv1_layer_call_fn_25295()*+MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ Ч
5__inference_batch_normalization_4_layer_call_fn_26006MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@я
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_25988MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
(__inference_bn_conv1_layer_call_fn_25369e()*+;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ 
p
Њ " џџџџџџџџџ у
B__inference_ResNet9_layer_call_and_return_conditional_losses_249632!"()*+89?@ABKLRSTUfgmnopyzЋЌ?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ		
p

 
Њ "%Ђ"

0џџџџџџџџџ
 
(__inference_bn_conv1_layer_call_fn_25378e()*+;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ 
p 
Њ " џџџџџџџџџ Ђ
5__inference_batch_normalization_4_layer_call_fn_26071i;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@
p
Њ " џџџџџџџџџ@Ђ
5__inference_batch_normalization_4_layer_call_fn_26080i;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@
p 
Њ " џџџџџџџџџ@У
-__inference_max_pooling2d_layer_call_fn_22885RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџУ
5__inference_batch_normalization_3_layer_call_fn_25827mnopMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@щ
N__inference_batch_normalization_layer_call_and_return_conditional_losses_25434?@ABMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 У
5__inference_batch_normalization_3_layer_call_fn_25836mnopMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@М
'__inference_ResNet9_layer_call_fn_244882!"()*+89?@ABKLRSTUfgmnopyzЋЌ@Ђ=
6Ђ3
)&
input_1џџџџџџџџџ		
p

 
Њ "џџџџџџџџџГ
G__inference_activation_2_layer_call_and_return_conditional_losses_25723h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ "-Ђ*
# 
0џџџџџџџџџ 
 
5__inference_batch_normalization_3_layer_call_fn_25901emnop;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@
p
Њ " џџџџџџџџџ@щ
N__inference_batch_normalization_layer_call_and_return_conditional_losses_25456?@ABMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
5__inference_batch_normalization_3_layer_call_fn_25910emnop;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@
p 
Њ " џџџџџџџџџ@Ф
N__inference_batch_normalization_layer_call_and_return_conditional_losses_25508r?@AB;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ 
p
Њ "-Ђ*
# 
0џџџџџџџџџ 
 
,__inference_activation_5_layer_call_fn_26090[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ " џџџџџџџџџ@Ф
N__inference_batch_normalization_layer_call_and_return_conditional_losses_25530r?@AB;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ 
p 
Њ "-Ђ*
# 
0џџџџџџџџџ 
 
B__inference_flatten_layer_call_and_return_conditional_losses_26278X/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "%Ђ"

0џџџџџџџџџ@
 м
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_23647RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ".Ђ+
$!
0џџџџџџџџџџџџџџџџџџ
 ж
A__inference_conv2d_layer_call_and_return_conditional_losses_2289689IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 Г
G__inference_activation_5_layer_call_and_return_conditional_losses_26085h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ "-Ђ*
# 
0џџџџџџџџџ@
 Г
G__inference_activation_3_layer_call_and_return_conditional_losses_25745h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ "-Ђ*
# 
0џџџџџџџџџ 
 М
'__inference_ResNet9_layer_call_fn_245982!"()*+89?@ABKLRSTUfgmnopyzЋЌ@Ђ=
6Ђ3
)&
input_1џџџџџџџџџ		
p 

 
Њ "џџџџџџџџџо
C__inference_bn_conv1_layer_call_and_return_conditional_losses_25264()*+MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
,__inference_activation_4_layer_call_fn_25920[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ " џџџџџџџџџ@Л
'__inference_ResNet9_layer_call_fn_252182!"()*+89?@ABKLRSTUfgmnopyzЋЌ?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ		
p 

 
Њ "џџџџџџџџџЛ
'__inference_ResNet9_layer_call_fn_251752!"()*+89?@ABKLRSTUfgmnopyzЋЌ?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ		
p

 
Њ "џџџџџџџџџо
C__inference_bn_conv1_layer_call_and_return_conditional_losses_25286()*+MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 Й
C__inference_bn_conv1_layer_call_and_return_conditional_losses_25338r()*+;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ 
p
Њ "-Ђ*
# 
0џџџџџџџџџ 
 А
(__inference_conv2d_3_layer_call_fn_23206fgIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@Й
C__inference_bn_conv1_layer_call_and_return_conditional_losses_25360r()*+;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ 
p 
Њ "-Ђ*
# 
0џџџџџџџџџ 
 ф
B__inference_ResNet9_layer_call_and_return_conditional_losses_243102!"()*+89?@ABKLRSTUfgmnopyzЋЌ@Ђ=
6Ђ3
)&
input_1џџџџџџџџџ		
p

 
Њ "%Ђ"

0џџџџџџџџџ
 Г
8__inference_global_average_pooling2d_layer_call_fn_23653wRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "!џџџџџџџџџџџџџџџџџџЦ
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_25604rRSTU;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ 
p
Њ "-Ђ*
# 
0џџџџџџџџџ 
 
,__inference_activation_1_layer_call_fn_25558[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ " џџџџџџџџџ Ц
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_25626rRSTU;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ 
p 
Њ "-Ђ*
# 
0џџџџџџџџџ 
 В
(__inference_conv2d_2_layer_call_fn_23508IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@Г
G__inference_activation_4_layer_call_and_return_conditional_losses_25915h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ "-Ђ*
# 
0џџџџџџџџџ@
 ы
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_25700RSTUMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 Г
G__inference_activation_6_layer_call_and_return_conditional_losses_26267h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ "-Ђ*
# 
0џџџџџџџџџ@
 и
C__inference_conv2d_1_layer_call_and_return_conditional_losses_23047KLIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 Ъ
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_26040v;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@
p
Њ "-Ђ*
# 
0џџџџџџџџџ@
 Ъ
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_26136v;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@
p
Њ "-Ђ*
# 
0џџџџџџџџџ@
 ф
B__inference_ResNet9_layer_call_and_return_conditional_losses_243772!"()*+89?@ABKLRSTUfgmnopyzЋЌ@Ђ=
6Ђ3
)&
input_1џџџџџџџџџ		
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 о
>__inference_add_layer_call_and_return_conditional_losses_25734jЂg
`Ђ]
[X
*'
inputs/0џџџџџџџџџ 
*'
inputs/1џџџџџџџџџ 
Њ "-Ђ*
# 
0џџџџџџџџџ 
 