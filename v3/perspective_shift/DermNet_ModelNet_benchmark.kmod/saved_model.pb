4
Ρ£
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
Ύ
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878Βέ,

conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:@*
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:@*
dtype0

batch_normalization_9/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_9/gamma

/batch_normalization_9/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_9/gamma*
_output_shapes
:@*
dtype0

batch_normalization_9/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_9/beta

.batch_normalization_9/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_9/beta*
_output_shapes
:@*
dtype0

!batch_normalization_9/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_9/moving_mean

5batch_normalization_9/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_9/moving_mean*
_output_shapes
:@*
dtype0
’
%batch_normalization_9/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_9/moving_variance

9batch_normalization_9/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_9/moving_variance*
_output_shapes
:@*
dtype0

conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
:@*
dtype0

batch_normalization_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_10/gamma

0batch_normalization_10/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_10/gamma*
_output_shapes
:@*
dtype0

batch_normalization_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_10/beta

/batch_normalization_10/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_10/beta*
_output_shapes
:@*
dtype0

"batch_normalization_10/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_10/moving_mean

6batch_normalization_10/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_10/moving_mean*
_output_shapes
:@*
dtype0
€
&batch_normalization_10/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_10/moving_variance

:batch_normalization_10/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_10/moving_variance*
_output_shapes
:@*
dtype0

conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_5/kernel
|
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*'
_output_shapes
:@*
dtype0
s
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_5/bias
l
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes	
:*
dtype0

batch_normalization_11/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_11/gamma

0batch_normalization_11/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_11/gamma*
_output_shapes	
:*
dtype0

batch_normalization_11/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_11/beta

/batch_normalization_11/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_11/beta*
_output_shapes	
:*
dtype0

"batch_normalization_11/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_11/moving_mean

6batch_normalization_11/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_11/moving_mean*
_output_shapes	
:*
dtype0
₯
&batch_normalization_11/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_11/moving_variance

:batch_normalization_11/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_11/moving_variance*
_output_shapes	
:*
dtype0
~
conv1d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv1d_2/kernel
w
#conv1d_2/kernel/Read/ReadVariableOpReadVariableOpconv1d_2/kernel*"
_output_shapes
:*
dtype0
r
conv1d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_2/bias
k
!conv1d_2/bias/Read/ReadVariableOpReadVariableOpconv1d_2/bias*
_output_shapes
:*
dtype0

batch_normalization_12/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_12/gamma

0batch_normalization_12/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_12/gamma*
_output_shapes
:*
dtype0

batch_normalization_12/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_12/beta

/batch_normalization_12/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_12/beta*
_output_shapes
:*
dtype0

"batch_normalization_12/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_12/moving_mean

6batch_normalization_12/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_12/moving_mean*
_output_shapes
:*
dtype0
€
&batch_normalization_12/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_12/moving_variance

:batch_normalization_12/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_12/moving_variance*
_output_shapes
:*
dtype0
x
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

:@*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:@*
dtype0

batch_normalization_13/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_13/gamma

0batch_normalization_13/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_13/gamma*
_output_shapes
:@*
dtype0

batch_normalization_13/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_13/beta

/batch_normalization_13/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_13/beta*
_output_shapes
:@*
dtype0

"batch_normalization_13/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_13/moving_mean

6batch_normalization_13/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_13/moving_mean*
_output_shapes
:@*
dtype0
€
&batch_normalization_13/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_13/moving_variance

:batch_normalization_13/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_13/moving_variance*
_output_shapes
:@*
dtype0
~
conv1d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv1d_3/kernel
w
#conv1d_3/kernel/Read/ReadVariableOpReadVariableOpconv1d_3/kernel*"
_output_shapes
:@*
dtype0
r
conv1d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_3/bias
k
!conv1d_3/bias/Read/ReadVariableOpReadVariableOpconv1d_3/bias*
_output_shapes
:*
dtype0

batch_normalization_14/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_14/gamma

0batch_normalization_14/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_14/gamma*
_output_shapes
:*
dtype0

batch_normalization_14/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_14/beta

/batch_normalization_14/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_14/beta*
_output_shapes
:*
dtype0

"batch_normalization_14/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_14/moving_mean

6batch_normalization_14/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_14/moving_mean*
_output_shapes
:*
dtype0
€
&batch_normalization_14/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_14/moving_variance

:batch_normalization_14/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_14/moving_variance*
_output_shapes
:*
dtype0
x
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_6/kernel
q
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes

:@*
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes
:@*
dtype0

batch_normalization_15/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_15/gamma

0batch_normalization_15/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_15/gamma*
_output_shapes
:@*
dtype0

batch_normalization_15/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_15/beta

/batch_normalization_15/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_15/beta*
_output_shapes
:@*
dtype0

"batch_normalization_15/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_15/moving_mean

6batch_normalization_15/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_15/moving_mean*
_output_shapes
:@*
dtype0
€
&batch_normalization_15/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_15/moving_variance

:batch_normalization_15/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_15/moving_variance*
_output_shapes
:@*
dtype0
z
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_7/kernel
s
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel* 
_output_shapes
:
*
dtype0
q
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_7/bias
j
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes	
:*
dtype0

batch_normalization_16/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_16/gamma

0batch_normalization_16/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_16/gamma*
_output_shapes	
:*
dtype0

batch_normalization_16/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_16/beta

/batch_normalization_16/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_16/beta*
_output_shapes	
:*
dtype0

"batch_normalization_16/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_16/moving_mean

6batch_normalization_16/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_16/moving_mean*
_output_shapes	
:*
dtype0
₯
&batch_normalization_16/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_16/moving_variance

:batch_normalization_16/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_16/moving_variance*
_output_shapes	
:*
dtype0
y
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*
shared_namedense_8/kernel
r
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes
:	@*
dtype0
p
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes
:@*
dtype0

batch_normalization_17/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_17/gamma

0batch_normalization_17/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_17/gamma*
_output_shapes
:@*
dtype0

batch_normalization_17/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_17/beta

/batch_normalization_17/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_17/beta*
_output_shapes
:@*
dtype0

"batch_normalization_17/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_17/moving_mean

6batch_normalization_17/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_17/moving_mean*
_output_shapes
:@*
dtype0
€
&batch_normalization_17/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_17/moving_variance

:batch_normalization_17/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_17/moving_variance*
_output_shapes
:@*
dtype0
x
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_9/kernel
q
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes

:@*
dtype0
p
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes
:*
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

Adam/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_3/kernel/m

*Adam/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/m*&
_output_shapes
:@*
dtype0

Adam/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_3/bias/m
y
(Adam/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/m*
_output_shapes
:@*
dtype0

"Adam/batch_normalization_9/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_9/gamma/m

6Adam/batch_normalization_9/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_9/gamma/m*
_output_shapes
:@*
dtype0

!Adam/batch_normalization_9/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_9/beta/m

5Adam/batch_normalization_9/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_9/beta/m*
_output_shapes
:@*
dtype0

Adam/conv2d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_4/kernel/m

*Adam/conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/m*&
_output_shapes
:@@*
dtype0

Adam/conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_4/bias/m
y
(Adam/conv2d_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/m*
_output_shapes
:@*
dtype0

#Adam/batch_normalization_10/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_10/gamma/m

7Adam/batch_normalization_10/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_10/gamma/m*
_output_shapes
:@*
dtype0

"Adam/batch_normalization_10/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_10/beta/m

6Adam/batch_normalization_10/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_10/beta/m*
_output_shapes
:@*
dtype0

Adam/conv2d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_5/kernel/m

*Adam/conv2d_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/m*'
_output_shapes
:@*
dtype0

Adam/conv2d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_5/bias/m
z
(Adam/conv2d_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/m*
_output_shapes	
:*
dtype0

#Adam/batch_normalization_11/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_11/gamma/m

7Adam/batch_normalization_11/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_11/gamma/m*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_11/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_11/beta/m

6Adam/batch_normalization_11/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_11/beta/m*
_output_shapes	
:*
dtype0

Adam/conv1d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv1d_2/kernel/m

*Adam/conv1d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_2/kernel/m*"
_output_shapes
:*
dtype0

Adam/conv1d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv1d_2/bias/m
y
(Adam/conv1d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_2/bias/m*
_output_shapes
:*
dtype0

#Adam/batch_normalization_12/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_12/gamma/m

7Adam/batch_normalization_12/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_12/gamma/m*
_output_shapes
:*
dtype0

"Adam/batch_normalization_12/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_12/beta/m

6Adam/batch_normalization_12/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_12/beta/m*
_output_shapes
:*
dtype0

Adam/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_5/kernel/m

)Adam/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/m*
_output_shapes

:@*
dtype0
~
Adam/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_5/bias/m
w
'Adam/dense_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/m*
_output_shapes
:@*
dtype0

#Adam/batch_normalization_13/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_13/gamma/m

7Adam/batch_normalization_13/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_13/gamma/m*
_output_shapes
:@*
dtype0

"Adam/batch_normalization_13/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_13/beta/m

6Adam/batch_normalization_13/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_13/beta/m*
_output_shapes
:@*
dtype0

Adam/conv1d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv1d_3/kernel/m

*Adam/conv1d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_3/kernel/m*"
_output_shapes
:@*
dtype0

Adam/conv1d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv1d_3/bias/m
y
(Adam/conv1d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_3/bias/m*
_output_shapes
:*
dtype0

#Adam/batch_normalization_14/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_14/gamma/m

7Adam/batch_normalization_14/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_14/gamma/m*
_output_shapes
:*
dtype0

"Adam/batch_normalization_14/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_14/beta/m

6Adam/batch_normalization_14/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_14/beta/m*
_output_shapes
:*
dtype0

Adam/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_6/kernel/m

)Adam/dense_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/m*
_output_shapes

:@*
dtype0
~
Adam/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_6/bias/m
w
'Adam/dense_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/m*
_output_shapes
:@*
dtype0

#Adam/batch_normalization_15/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_15/gamma/m

7Adam/batch_normalization_15/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_15/gamma/m*
_output_shapes
:@*
dtype0

"Adam/batch_normalization_15/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_15/beta/m

6Adam/batch_normalization_15/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_15/beta/m*
_output_shapes
:@*
dtype0

Adam/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_7/kernel/m

)Adam/dense_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_7/bias/m
x
'Adam/dense_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/m*
_output_shapes	
:*
dtype0

#Adam/batch_normalization_16/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_16/gamma/m

7Adam/batch_normalization_16/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_16/gamma/m*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_16/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_16/beta/m

6Adam/batch_normalization_16/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_16/beta/m*
_output_shapes	
:*
dtype0

Adam/dense_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*&
shared_nameAdam/dense_8/kernel/m

)Adam/dense_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/m*
_output_shapes
:	@*
dtype0
~
Adam/dense_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_8/bias/m
w
'Adam/dense_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/m*
_output_shapes
:@*
dtype0

#Adam/batch_normalization_17/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_17/gamma/m

7Adam/batch_normalization_17/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_17/gamma/m*
_output_shapes
:@*
dtype0

"Adam/batch_normalization_17/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_17/beta/m

6Adam/batch_normalization_17/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_17/beta/m*
_output_shapes
:@*
dtype0

Adam/dense_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_9/kernel/m

)Adam/dense_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/m*
_output_shapes

:@*
dtype0
~
Adam/dense_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_9/bias/m
w
'Adam/dense_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_3/kernel/v

*Adam/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/v*&
_output_shapes
:@*
dtype0

Adam/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_3/bias/v
y
(Adam/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/v*
_output_shapes
:@*
dtype0

"Adam/batch_normalization_9/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_9/gamma/v

6Adam/batch_normalization_9/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_9/gamma/v*
_output_shapes
:@*
dtype0

!Adam/batch_normalization_9/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_9/beta/v

5Adam/batch_normalization_9/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_9/beta/v*
_output_shapes
:@*
dtype0

Adam/conv2d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_4/kernel/v

*Adam/conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/v*&
_output_shapes
:@@*
dtype0

Adam/conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_4/bias/v
y
(Adam/conv2d_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/v*
_output_shapes
:@*
dtype0

#Adam/batch_normalization_10/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_10/gamma/v

7Adam/batch_normalization_10/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_10/gamma/v*
_output_shapes
:@*
dtype0

"Adam/batch_normalization_10/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_10/beta/v

6Adam/batch_normalization_10/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_10/beta/v*
_output_shapes
:@*
dtype0

Adam/conv2d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_5/kernel/v

*Adam/conv2d_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/v*'
_output_shapes
:@*
dtype0

Adam/conv2d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_5/bias/v
z
(Adam/conv2d_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/v*
_output_shapes	
:*
dtype0

#Adam/batch_normalization_11/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_11/gamma/v

7Adam/batch_normalization_11/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_11/gamma/v*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_11/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_11/beta/v

6Adam/batch_normalization_11/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_11/beta/v*
_output_shapes	
:*
dtype0

Adam/conv1d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv1d_2/kernel/v

*Adam/conv1d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_2/kernel/v*"
_output_shapes
:*
dtype0

Adam/conv1d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv1d_2/bias/v
y
(Adam/conv1d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_2/bias/v*
_output_shapes
:*
dtype0

#Adam/batch_normalization_12/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_12/gamma/v

7Adam/batch_normalization_12/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_12/gamma/v*
_output_shapes
:*
dtype0

"Adam/batch_normalization_12/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_12/beta/v

6Adam/batch_normalization_12/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_12/beta/v*
_output_shapes
:*
dtype0

Adam/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_5/kernel/v

)Adam/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/v*
_output_shapes

:@*
dtype0
~
Adam/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_5/bias/v
w
'Adam/dense_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/v*
_output_shapes
:@*
dtype0

#Adam/batch_normalization_13/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_13/gamma/v

7Adam/batch_normalization_13/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_13/gamma/v*
_output_shapes
:@*
dtype0

"Adam/batch_normalization_13/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_13/beta/v

6Adam/batch_normalization_13/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_13/beta/v*
_output_shapes
:@*
dtype0

Adam/conv1d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv1d_3/kernel/v

*Adam/conv1d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_3/kernel/v*"
_output_shapes
:@*
dtype0

Adam/conv1d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv1d_3/bias/v
y
(Adam/conv1d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_3/bias/v*
_output_shapes
:*
dtype0

#Adam/batch_normalization_14/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_14/gamma/v

7Adam/batch_normalization_14/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_14/gamma/v*
_output_shapes
:*
dtype0

"Adam/batch_normalization_14/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_14/beta/v

6Adam/batch_normalization_14/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_14/beta/v*
_output_shapes
:*
dtype0

Adam/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_6/kernel/v

)Adam/dense_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/v*
_output_shapes

:@*
dtype0
~
Adam/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_6/bias/v
w
'Adam/dense_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/v*
_output_shapes
:@*
dtype0

#Adam/batch_normalization_15/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_15/gamma/v

7Adam/batch_normalization_15/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_15/gamma/v*
_output_shapes
:@*
dtype0

"Adam/batch_normalization_15/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_15/beta/v

6Adam/batch_normalization_15/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_15/beta/v*
_output_shapes
:@*
dtype0

Adam/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_7/kernel/v

)Adam/dense_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_7/bias/v
x
'Adam/dense_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/v*
_output_shapes	
:*
dtype0

#Adam/batch_normalization_16/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_16/gamma/v

7Adam/batch_normalization_16/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_16/gamma/v*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_16/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_16/beta/v

6Adam/batch_normalization_16/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_16/beta/v*
_output_shapes	
:*
dtype0

Adam/dense_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*&
shared_nameAdam/dense_8/kernel/v

)Adam/dense_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/v*
_output_shapes
:	@*
dtype0
~
Adam/dense_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_8/bias/v
w
'Adam/dense_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/v*
_output_shapes
:@*
dtype0

#Adam/batch_normalization_17/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_17/gamma/v

7Adam/batch_normalization_17/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_17/gamma/v*
_output_shapes
:@*
dtype0

"Adam/batch_normalization_17/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_17/beta/v

6Adam/batch_normalization_17/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_17/beta/v*
_output_shapes
:@*
dtype0

Adam/dense_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_9/kernel/v

)Adam/dense_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/v*
_output_shapes

:@*
dtype0
~
Adam/dense_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_9/bias/v
w
'Adam/dense_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
Ιΰ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ΰ
valueψίBτί Bμί
Κ
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer-9
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer_with_weights-8
layer-13
layer_with_weights-9
layer-14
layer_with_weights-10
layer-15
layer_with_weights-11
layer-16
layer_with_weights-12
layer-17
layer_with_weights-13
layer-18
layer-19
layer_with_weights-14
layer-20
layer_with_weights-15
layer-21
layer_with_weights-16
layer-22
layer_with_weights-17
layer-23
layer_with_weights-18
layer-24
layer-25
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api
 
signatures
 
R
!regularization_losses
"trainable_variables
#	variables
$	keras_api
h

%kernel
&bias
'regularization_losses
(trainable_variables
)	variables
*	keras_api

+axis
	,gamma
-beta
.moving_mean
/moving_variance
0regularization_losses
1trainable_variables
2	variables
3	keras_api
h

4kernel
5bias
6regularization_losses
7trainable_variables
8	variables
9	keras_api

:axis
	;gamma
<beta
=moving_mean
>moving_variance
?regularization_losses
@trainable_variables
A	variables
B	keras_api
h

Ckernel
Dbias
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api

Iaxis
	Jgamma
Kbeta
Lmoving_mean
Mmoving_variance
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
R
Rregularization_losses
Strainable_variables
T	variables
U	keras_api
R
Vregularization_losses
Wtrainable_variables
X	variables
Y	keras_api
R
Zregularization_losses
[trainable_variables
\	variables
]	keras_api
h

^kernel
_bias
`regularization_losses
atrainable_variables
b	variables
c	keras_api

daxis
	egamma
fbeta
gmoving_mean
hmoving_variance
iregularization_losses
jtrainable_variables
k	variables
l	keras_api
h

mkernel
nbias
oregularization_losses
ptrainable_variables
q	variables
r	keras_api

saxis
	tgamma
ubeta
vmoving_mean
wmoving_variance
xregularization_losses
ytrainable_variables
z	variables
{	keras_api
j

|kernel
}bias
~regularization_losses
trainable_variables
	variables
	keras_api
 
	axis

gamma
	beta
moving_mean
moving_variance
regularization_losses
trainable_variables
	variables
	keras_api
n
kernel
	bias
regularization_losses
trainable_variables
	variables
	keras_api
 
	axis

gamma
	beta
moving_mean
moving_variance
regularization_losses
trainable_variables
	variables
	keras_api
V
regularization_losses
trainable_variables
	variables
	keras_api
n
kernel
	bias
 regularization_losses
‘trainable_variables
’	variables
£	keras_api
 
	€axis

₯gamma
	¦beta
§moving_mean
¨moving_variance
©regularization_losses
ͺtrainable_variables
«	variables
¬	keras_api
n
­kernel
	?bias
―regularization_losses
°trainable_variables
±	variables
²	keras_api
 
	³axis

΄gamma
	΅beta
Άmoving_mean
·moving_variance
Έregularization_losses
Ήtrainable_variables
Ί	variables
»	keras_api
n
Όkernel
	½bias
Ύregularization_losses
Ώtrainable_variables
ΐ	variables
Α	keras_api
V
Βregularization_losses
Γtrainable_variables
Δ	variables
Ε	keras_api
έ
	Ζiter
Ηbeta_1
Θbeta_2

Ιdecay
Κlearning_rate%m?&mΣ,mΤ-mΥ4mΦ5mΧ;mΨ<mΩCmΪDmΫJmάKmέ^mή_mίemΰfmαmmβnmγtmδumε|mζ}mη	mθ	mι	mκ	mλ	mμ	mν	mξ	mο	₯mπ	¦mρ	­mς	?mσ	΄mτ	΅mυ	Όmφ	½mχ%vψ&vω,vϊ-vϋ4vό5vύ;vώ<v?CvDvJvKv^v_vevfvmvnvtvuv|v}v	v	v	v	v	v	v	v	v	₯v	¦v	­v	?v	΄v	΅v	Όv	½v
 
Ά
%0
&1
,2
-3
44
55
;6
<7
C8
D9
J10
K11
^12
_13
e14
f15
m16
n17
t18
u19
|20
}21
22
23
24
25
26
27
28
29
₯30
¦31
­32
?33
΄34
΅35
Ό36
½37
Ξ
%0
&1
,2
-3
.4
/5
46
57
;8
<9
=10
>11
C12
D13
J14
K15
L16
M17
^18
_19
e20
f21
g22
h23
m24
n25
t26
u27
v28
w29
|30
}31
32
33
34
35
36
37
38
39
40
41
42
43
₯44
¦45
§46
¨47
­48
?49
΄50
΅51
Ά52
·53
Ό54
½55
²
Λlayer_metrics
 Μlayer_regularization_losses
Νlayers
Ξnon_trainable_variables
regularization_losses
trainable_variables
	variables
Οmetrics
 
 
 
 
²
Πlayer_metrics
 Ρlayer_regularization_losses
?layers
Σnon_trainable_variables
!regularization_losses
"trainable_variables
#	variables
Τmetrics
[Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

%0
&1

%0
&1
²
Υlayer_metrics
 Φlayer_regularization_losses
Χlayers
Ψnon_trainable_variables
'regularization_losses
(trainable_variables
)	variables
Ωmetrics
 
fd
VARIABLE_VALUEbatch_normalization_9/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_9/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_9/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_9/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

,0
-1

,0
-1
.2
/3
²
Ϊlayer_metrics
 Ϋlayer_regularization_losses
άlayers
έnon_trainable_variables
0regularization_losses
1trainable_variables
2	variables
ήmetrics
[Y
VARIABLE_VALUEconv2d_4/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_4/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

40
51

40
51
²
ίlayer_metrics
 ΰlayer_regularization_losses
αlayers
βnon_trainable_variables
6regularization_losses
7trainable_variables
8	variables
γmetrics
 
ge
VARIABLE_VALUEbatch_normalization_10/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_10/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_10/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_10/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

;0
<1

;0
<1
=2
>3
²
δlayer_metrics
 εlayer_regularization_losses
ζlayers
ηnon_trainable_variables
?regularization_losses
@trainable_variables
A	variables
θmetrics
[Y
VARIABLE_VALUEconv2d_5/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_5/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

C0
D1

C0
D1
²
ιlayer_metrics
 κlayer_regularization_losses
λlayers
μnon_trainable_variables
Eregularization_losses
Ftrainable_variables
G	variables
νmetrics
 
ge
VARIABLE_VALUEbatch_normalization_11/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_11/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_11/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_11/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

J0
K1

J0
K1
L2
M3
²
ξlayer_metrics
 οlayer_regularization_losses
πlayers
ρnon_trainable_variables
Nregularization_losses
Otrainable_variables
P	variables
ςmetrics
 
 
 
²
σlayer_metrics
 τlayer_regularization_losses
υlayers
φnon_trainable_variables
Rregularization_losses
Strainable_variables
T	variables
χmetrics
 
 
 
²
ψlayer_metrics
 ωlayer_regularization_losses
ϊlayers
ϋnon_trainable_variables
Vregularization_losses
Wtrainable_variables
X	variables
όmetrics
 
 
 
²
ύlayer_metrics
 ώlayer_regularization_losses
?layers
non_trainable_variables
Zregularization_losses
[trainable_variables
\	variables
metrics
[Y
VARIABLE_VALUEconv1d_2/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_2/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

^0
_1

^0
_1
²
layer_metrics
 layer_regularization_losses
layers
non_trainable_variables
`regularization_losses
atrainable_variables
b	variables
metrics
 
ge
VARIABLE_VALUEbatch_normalization_12/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_12/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_12/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_12/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

e0
f1

e0
f1
g2
h3
²
layer_metrics
 layer_regularization_losses
layers
non_trainable_variables
iregularization_losses
jtrainable_variables
k	variables
metrics
ZX
VARIABLE_VALUEdense_5/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_5/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

m0
n1

m0
n1
²
layer_metrics
 layer_regularization_losses
layers
non_trainable_variables
oregularization_losses
ptrainable_variables
q	variables
metrics
 
ge
VARIABLE_VALUEbatch_normalization_13/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_13/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_13/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_13/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

t0
u1

t0
u1
v2
w3
²
layer_metrics
 layer_regularization_losses
layers
non_trainable_variables
xregularization_losses
ytrainable_variables
z	variables
metrics
\Z
VARIABLE_VALUEconv1d_3/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_3/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
 

|0
}1

|0
}1
³
layer_metrics
 layer_regularization_losses
layers
non_trainable_variables
~regularization_losses
trainable_variables
	variables
metrics
 
hf
VARIABLE_VALUEbatch_normalization_14/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_14/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_14/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_14/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
 
0
1
2
3
΅
layer_metrics
 layer_regularization_losses
layers
non_trainable_variables
regularization_losses
trainable_variables
	variables
metrics
[Y
VARIABLE_VALUEdense_6/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_6/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
΅
 layer_metrics
 ‘layer_regularization_losses
’layers
£non_trainable_variables
regularization_losses
trainable_variables
	variables
€metrics
 
hf
VARIABLE_VALUEbatch_normalization_15/gamma6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_15/beta5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_15/moving_mean<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_15/moving_variance@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
 
0
1
2
3
΅
₯layer_metrics
 ¦layer_regularization_losses
§layers
¨non_trainable_variables
regularization_losses
trainable_variables
	variables
©metrics
 
 
 
΅
ͺlayer_metrics
 «layer_regularization_losses
¬layers
­non_trainable_variables
regularization_losses
trainable_variables
	variables
?metrics
[Y
VARIABLE_VALUEdense_7/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_7/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
΅
―layer_metrics
 °layer_regularization_losses
±layers
²non_trainable_variables
 regularization_losses
‘trainable_variables
’	variables
³metrics
 
hf
VARIABLE_VALUEbatch_normalization_16/gamma6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_16/beta5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_16/moving_mean<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_16/moving_variance@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

₯0
¦1
 
₯0
¦1
§2
¨3
΅
΄layer_metrics
 ΅layer_regularization_losses
Άlayers
·non_trainable_variables
©regularization_losses
ͺtrainable_variables
«	variables
Έmetrics
[Y
VARIABLE_VALUEdense_8/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_8/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE
 

­0
?1

­0
?1
΅
Ήlayer_metrics
 Ίlayer_regularization_losses
»layers
Όnon_trainable_variables
―regularization_losses
°trainable_variables
±	variables
½metrics
 
hf
VARIABLE_VALUEbatch_normalization_17/gamma6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_17/beta5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_17/moving_mean<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_17/moving_variance@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

΄0
΅1
 
΄0
΅1
Ά2
·3
΅
Ύlayer_metrics
 Ώlayer_regularization_losses
ΐlayers
Αnon_trainable_variables
Έregularization_losses
Ήtrainable_variables
Ί	variables
Βmetrics
[Y
VARIABLE_VALUEdense_9/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_9/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Ό0
½1

Ό0
½1
΅
Γlayer_metrics
 Δlayer_regularization_losses
Εlayers
Ζnon_trainable_variables
Ύregularization_losses
Ώtrainable_variables
ΐ	variables
Ηmetrics
 
 
 
΅
Θlayer_metrics
 Ιlayer_regularization_losses
Κlayers
Λnon_trainable_variables
Βregularization_losses
Γtrainable_variables
Δ	variables
Μmetrics
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
 
 
Ζ
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

.0
/1
=2
>3
L4
M5
g6
h7
v8
w9
10
11
12
13
§14
¨15
Ά16
·17

Ν0
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
.0
/1
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
=0
>1
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
L0
M1
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
 
 
 
 

g0
h1
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
v0
w1
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
0
1
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
0
1
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

§0
¨1
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
Ά0
·1
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
8

Ξtotal

Οcount
Π	variables
Ρ	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

Ξ0
Ο1

Π	variables
~|
VARIABLE_VALUEAdam/conv2d_3/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_3/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_9/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_9/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_4/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_4/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_10/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_10/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_5/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_5/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_11/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_11/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv1d_2/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv1d_2/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_12/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_12/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_5/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_5/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_13/gamma/mQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_13/beta/mPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_3/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_3/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_14/gamma/mRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_14/beta/mQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_6/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_6/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_15/gamma/mRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_15/beta/mQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_7/kernel/mSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_7/bias/mQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_16/gamma/mRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_16/beta/mQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_8/kernel/mSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_8/bias/mQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_17/gamma/mRlayer_with_weights-17/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_17/beta/mQlayer_with_weights-17/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_9/kernel/mSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_9/bias/mQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_3/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_3/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_9/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_9/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_4/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_4/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_10/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_10/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_5/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_5/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_11/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_11/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv1d_2/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv1d_2/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_12/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_12/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_5/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_5/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_13/gamma/vQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_13/beta/vPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_3/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_3/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_14/gamma/vRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_14/beta/vQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_6/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_6/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_15/gamma/vRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_15/beta/vQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_7/kernel/vSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_7/bias/vQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_16/gamma/vRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_16/beta/vQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_8/kernel/vSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_8/bias/vQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_17/gamma/vRlayer_with_weights-17/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_17/beta/vQlayer_with_weights-17/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_9/kernel/vSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_9/bias/vQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_2Placeholder*,
_output_shapes
:?????????*
dtype0*!
shape:?????????

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2conv2d_3/kernelconv2d_3/biasbatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_varianceconv2d_4/kernelconv2d_4/biasbatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_varianceconv2d_5/kernelconv2d_5/biasbatch_normalization_11/gammabatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_varianceconv1d_2/kernelconv1d_2/bias&batch_normalization_12/moving_variancebatch_normalization_12/gamma"batch_normalization_12/moving_meanbatch_normalization_12/betadense_5/kerneldense_5/bias&batch_normalization_13/moving_variancebatch_normalization_13/gamma"batch_normalization_13/moving_meanbatch_normalization_13/betaconv1d_3/kernelconv1d_3/bias&batch_normalization_14/moving_variancebatch_normalization_14/gamma"batch_normalization_14/moving_meanbatch_normalization_14/betadense_6/kerneldense_6/bias&batch_normalization_15/moving_variancebatch_normalization_15/gamma"batch_normalization_15/moving_meanbatch_normalization_15/betadense_7/kerneldense_7/bias&batch_normalization_16/moving_variancebatch_normalization_16/gamma"batch_normalization_16/moving_meanbatch_normalization_16/betadense_8/kerneldense_8/bias&batch_normalization_17/moving_variancebatch_normalization_17/gamma"batch_normalization_17/moving_meanbatch_normalization_17/betadense_9/kerneldense_9/bias*D
Tin=
;29*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*Z
_read_only_resource_inputs<
:8	
 !"#$%&'()*+,-./012345678*2
config_proto" 

CPU

GPU2 *0J 8 *-
f(R&
$__inference_signature_wrapper_400707
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ρ6
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp/batch_normalization_9/gamma/Read/ReadVariableOp.batch_normalization_9/beta/Read/ReadVariableOp5batch_normalization_9/moving_mean/Read/ReadVariableOp9batch_normalization_9/moving_variance/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp0batch_normalization_10/gamma/Read/ReadVariableOp/batch_normalization_10/beta/Read/ReadVariableOp6batch_normalization_10/moving_mean/Read/ReadVariableOp:batch_normalization_10/moving_variance/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp0batch_normalization_11/gamma/Read/ReadVariableOp/batch_normalization_11/beta/Read/ReadVariableOp6batch_normalization_11/moving_mean/Read/ReadVariableOp:batch_normalization_11/moving_variance/Read/ReadVariableOp#conv1d_2/kernel/Read/ReadVariableOp!conv1d_2/bias/Read/ReadVariableOp0batch_normalization_12/gamma/Read/ReadVariableOp/batch_normalization_12/beta/Read/ReadVariableOp6batch_normalization_12/moving_mean/Read/ReadVariableOp:batch_normalization_12/moving_variance/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp0batch_normalization_13/gamma/Read/ReadVariableOp/batch_normalization_13/beta/Read/ReadVariableOp6batch_normalization_13/moving_mean/Read/ReadVariableOp:batch_normalization_13/moving_variance/Read/ReadVariableOp#conv1d_3/kernel/Read/ReadVariableOp!conv1d_3/bias/Read/ReadVariableOp0batch_normalization_14/gamma/Read/ReadVariableOp/batch_normalization_14/beta/Read/ReadVariableOp6batch_normalization_14/moving_mean/Read/ReadVariableOp:batch_normalization_14/moving_variance/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp0batch_normalization_15/gamma/Read/ReadVariableOp/batch_normalization_15/beta/Read/ReadVariableOp6batch_normalization_15/moving_mean/Read/ReadVariableOp:batch_normalization_15/moving_variance/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOp0batch_normalization_16/gamma/Read/ReadVariableOp/batch_normalization_16/beta/Read/ReadVariableOp6batch_normalization_16/moving_mean/Read/ReadVariableOp:batch_normalization_16/moving_variance/Read/ReadVariableOp"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOp0batch_normalization_17/gamma/Read/ReadVariableOp/batch_normalization_17/beta/Read/ReadVariableOp6batch_normalization_17/moving_mean/Read/ReadVariableOp:batch_normalization_17/moving_variance/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/conv2d_3/kernel/m/Read/ReadVariableOp(Adam/conv2d_3/bias/m/Read/ReadVariableOp6Adam/batch_normalization_9/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_9/beta/m/Read/ReadVariableOp*Adam/conv2d_4/kernel/m/Read/ReadVariableOp(Adam/conv2d_4/bias/m/Read/ReadVariableOp7Adam/batch_normalization_10/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_10/beta/m/Read/ReadVariableOp*Adam/conv2d_5/kernel/m/Read/ReadVariableOp(Adam/conv2d_5/bias/m/Read/ReadVariableOp7Adam/batch_normalization_11/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_11/beta/m/Read/ReadVariableOp*Adam/conv1d_2/kernel/m/Read/ReadVariableOp(Adam/conv1d_2/bias/m/Read/ReadVariableOp7Adam/batch_normalization_12/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_12/beta/m/Read/ReadVariableOp)Adam/dense_5/kernel/m/Read/ReadVariableOp'Adam/dense_5/bias/m/Read/ReadVariableOp7Adam/batch_normalization_13/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_13/beta/m/Read/ReadVariableOp*Adam/conv1d_3/kernel/m/Read/ReadVariableOp(Adam/conv1d_3/bias/m/Read/ReadVariableOp7Adam/batch_normalization_14/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_14/beta/m/Read/ReadVariableOp)Adam/dense_6/kernel/m/Read/ReadVariableOp'Adam/dense_6/bias/m/Read/ReadVariableOp7Adam/batch_normalization_15/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_15/beta/m/Read/ReadVariableOp)Adam/dense_7/kernel/m/Read/ReadVariableOp'Adam/dense_7/bias/m/Read/ReadVariableOp7Adam/batch_normalization_16/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_16/beta/m/Read/ReadVariableOp)Adam/dense_8/kernel/m/Read/ReadVariableOp'Adam/dense_8/bias/m/Read/ReadVariableOp7Adam/batch_normalization_17/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_17/beta/m/Read/ReadVariableOp)Adam/dense_9/kernel/m/Read/ReadVariableOp'Adam/dense_9/bias/m/Read/ReadVariableOp*Adam/conv2d_3/kernel/v/Read/ReadVariableOp(Adam/conv2d_3/bias/v/Read/ReadVariableOp6Adam/batch_normalization_9/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_9/beta/v/Read/ReadVariableOp*Adam/conv2d_4/kernel/v/Read/ReadVariableOp(Adam/conv2d_4/bias/v/Read/ReadVariableOp7Adam/batch_normalization_10/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_10/beta/v/Read/ReadVariableOp*Adam/conv2d_5/kernel/v/Read/ReadVariableOp(Adam/conv2d_5/bias/v/Read/ReadVariableOp7Adam/batch_normalization_11/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_11/beta/v/Read/ReadVariableOp*Adam/conv1d_2/kernel/v/Read/ReadVariableOp(Adam/conv1d_2/bias/v/Read/ReadVariableOp7Adam/batch_normalization_12/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_12/beta/v/Read/ReadVariableOp)Adam/dense_5/kernel/v/Read/ReadVariableOp'Adam/dense_5/bias/v/Read/ReadVariableOp7Adam/batch_normalization_13/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_13/beta/v/Read/ReadVariableOp*Adam/conv1d_3/kernel/v/Read/ReadVariableOp(Adam/conv1d_3/bias/v/Read/ReadVariableOp7Adam/batch_normalization_14/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_14/beta/v/Read/ReadVariableOp)Adam/dense_6/kernel/v/Read/ReadVariableOp'Adam/dense_6/bias/v/Read/ReadVariableOp7Adam/batch_normalization_15/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_15/beta/v/Read/ReadVariableOp)Adam/dense_7/kernel/v/Read/ReadVariableOp'Adam/dense_7/bias/v/Read/ReadVariableOp7Adam/batch_normalization_16/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_16/beta/v/Read/ReadVariableOp)Adam/dense_8/kernel/v/Read/ReadVariableOp'Adam/dense_8/bias/v/Read/ReadVariableOp7Adam/batch_normalization_17/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_17/beta/v/Read/ReadVariableOp)Adam/dense_9/kernel/v/Read/ReadVariableOp'Adam/dense_9/bias/v/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *(
f#R!
__inference__traced_save_403544
!
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_3/kernelconv2d_3/biasbatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_varianceconv2d_4/kernelconv2d_4/biasbatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_varianceconv2d_5/kernelconv2d_5/biasbatch_normalization_11/gammabatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_varianceconv1d_2/kernelconv1d_2/biasbatch_normalization_12/gammabatch_normalization_12/beta"batch_normalization_12/moving_mean&batch_normalization_12/moving_variancedense_5/kerneldense_5/biasbatch_normalization_13/gammabatch_normalization_13/beta"batch_normalization_13/moving_mean&batch_normalization_13/moving_varianceconv1d_3/kernelconv1d_3/biasbatch_normalization_14/gammabatch_normalization_14/beta"batch_normalization_14/moving_mean&batch_normalization_14/moving_variancedense_6/kerneldense_6/biasbatch_normalization_15/gammabatch_normalization_15/beta"batch_normalization_15/moving_mean&batch_normalization_15/moving_variancedense_7/kerneldense_7/biasbatch_normalization_16/gammabatch_normalization_16/beta"batch_normalization_16/moving_mean&batch_normalization_16/moving_variancedense_8/kerneldense_8/biasbatch_normalization_17/gammabatch_normalization_17/beta"batch_normalization_17/moving_mean&batch_normalization_17/moving_variancedense_9/kerneldense_9/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv2d_3/kernel/mAdam/conv2d_3/bias/m"Adam/batch_normalization_9/gamma/m!Adam/batch_normalization_9/beta/mAdam/conv2d_4/kernel/mAdam/conv2d_4/bias/m#Adam/batch_normalization_10/gamma/m"Adam/batch_normalization_10/beta/mAdam/conv2d_5/kernel/mAdam/conv2d_5/bias/m#Adam/batch_normalization_11/gamma/m"Adam/batch_normalization_11/beta/mAdam/conv1d_2/kernel/mAdam/conv1d_2/bias/m#Adam/batch_normalization_12/gamma/m"Adam/batch_normalization_12/beta/mAdam/dense_5/kernel/mAdam/dense_5/bias/m#Adam/batch_normalization_13/gamma/m"Adam/batch_normalization_13/beta/mAdam/conv1d_3/kernel/mAdam/conv1d_3/bias/m#Adam/batch_normalization_14/gamma/m"Adam/batch_normalization_14/beta/mAdam/dense_6/kernel/mAdam/dense_6/bias/m#Adam/batch_normalization_15/gamma/m"Adam/batch_normalization_15/beta/mAdam/dense_7/kernel/mAdam/dense_7/bias/m#Adam/batch_normalization_16/gamma/m"Adam/batch_normalization_16/beta/mAdam/dense_8/kernel/mAdam/dense_8/bias/m#Adam/batch_normalization_17/gamma/m"Adam/batch_normalization_17/beta/mAdam/dense_9/kernel/mAdam/dense_9/bias/mAdam/conv2d_3/kernel/vAdam/conv2d_3/bias/v"Adam/batch_normalization_9/gamma/v!Adam/batch_normalization_9/beta/vAdam/conv2d_4/kernel/vAdam/conv2d_4/bias/v#Adam/batch_normalization_10/gamma/v"Adam/batch_normalization_10/beta/vAdam/conv2d_5/kernel/vAdam/conv2d_5/bias/v#Adam/batch_normalization_11/gamma/v"Adam/batch_normalization_11/beta/vAdam/conv1d_2/kernel/vAdam/conv1d_2/bias/v#Adam/batch_normalization_12/gamma/v"Adam/batch_normalization_12/beta/vAdam/dense_5/kernel/vAdam/dense_5/bias/v#Adam/batch_normalization_13/gamma/v"Adam/batch_normalization_13/beta/vAdam/conv1d_3/kernel/vAdam/conv1d_3/bias/v#Adam/batch_normalization_14/gamma/v"Adam/batch_normalization_14/beta/vAdam/dense_6/kernel/vAdam/dense_6/bias/v#Adam/batch_normalization_15/gamma/v"Adam/batch_normalization_15/beta/vAdam/dense_7/kernel/vAdam/dense_7/bias/v#Adam/batch_normalization_16/gamma/v"Adam/batch_normalization_16/beta/vAdam/dense_8/kernel/vAdam/dense_8/bias/v#Adam/batch_normalization_17/gamma/v"Adam/batch_normalization_17/beta/vAdam/dense_9/kernel/vAdam/dense_9/bias/v*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *+
f&R$
"__inference__traced_restore_403971Έδ'
Σ
U
9__inference_tf_op_layer_ExpandDims_1_layer_call_fn_401606

inputs
identityΰ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_tf_op_layer_ExpandDims_1_layer_call_and_return_conditional_losses_3988832
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:?????????:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
*
Λ
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_398549

inputs
assignmovingavg_398524
assignmovingavg_1_398530)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity’#AssignMovingAvg/AssignSubVariableOp’%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????@2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesΆ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze_1
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/398524*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_398524*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpΓ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/398524*
_output_shapes
:@2
AssignMovingAvg/subΊ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/398524*
_output_shapes
:@2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_398524AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/398524*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp€
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/398530*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_398530*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpΝ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/398530*
_output_shapes
:@2
AssignMovingAvg_1/subΔ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/398530*
_output_shapes
:@2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_398530AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/398530*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????@2
batchnorm/add_1Β
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*4
_output_shapes"
 :??????????????????@2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs
Ρ
R
6__inference_tf_op_layer_Reshape_1_layer_call_fn_402061

inputs
identityΪ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_3991972
PartitionedCallr
IdentityIdentityPartitionedCall:output:0*
T0*-
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:?????????:Y U
1
_output_shapes
:?????????
 
_user_specified_nameinputs
«
«
C__inference_dense_8_layer_call_and_return_conditional_losses_402982

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:::P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
Ξ
ͺ
7__inference_batch_normalization_14_layer_call_fn_402654

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall₯
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_3995682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs

ϋ
H__inference_functional_3_layer_call_and_return_conditional_losses_400063
input_2
conv2d_3_399926
conv2d_3_399928 
batch_normalization_9_399931 
batch_normalization_9_399933 
batch_normalization_9_399935 
batch_normalization_9_399937
conv2d_4_399940
conv2d_4_399942!
batch_normalization_10_399945!
batch_normalization_10_399947!
batch_normalization_10_399949!
batch_normalization_10_399951
conv2d_5_399954
conv2d_5_399956!
batch_normalization_11_399959!
batch_normalization_11_399961!
batch_normalization_11_399963!
batch_normalization_11_399965
conv1d_2_399971
conv1d_2_399973!
batch_normalization_12_399976!
batch_normalization_12_399978!
batch_normalization_12_399980!
batch_normalization_12_399982
dense_5_399985
dense_5_399987!
batch_normalization_13_399990!
batch_normalization_13_399992!
batch_normalization_13_399994!
batch_normalization_13_399996
conv1d_3_399999
conv1d_3_400001!
batch_normalization_14_400004!
batch_normalization_14_400006!
batch_normalization_14_400008!
batch_normalization_14_400010
dense_6_400013
dense_6_400015!
batch_normalization_15_400018!
batch_normalization_15_400020!
batch_normalization_15_400022!
batch_normalization_15_400024
dense_7_400028
dense_7_400030!
batch_normalization_16_400033!
batch_normalization_16_400035!
batch_normalization_16_400037!
batch_normalization_16_400039
dense_8_400042
dense_8_400044!
batch_normalization_17_400047!
batch_normalization_17_400049!
batch_normalization_17_400051!
batch_normalization_17_400053
dense_9_400056
dense_9_400058
identity’.batch_normalization_10/StatefulPartitionedCall’.batch_normalization_11/StatefulPartitionedCall’.batch_normalization_12/StatefulPartitionedCall’.batch_normalization_13/StatefulPartitionedCall’.batch_normalization_14/StatefulPartitionedCall’.batch_normalization_15/StatefulPartitionedCall’.batch_normalization_16/StatefulPartitionedCall’.batch_normalization_17/StatefulPartitionedCall’-batch_normalization_9/StatefulPartitionedCall’ conv1d_2/StatefulPartitionedCall’ conv1d_3/StatefulPartitionedCall’ conv2d_3/StatefulPartitionedCall’ conv2d_4/StatefulPartitionedCall’ conv2d_5/StatefulPartitionedCall’dense_5/StatefulPartitionedCall’dense_6/StatefulPartitionedCall’dense_7/StatefulPartitionedCall’dense_8/StatefulPartitionedCall’dense_9/StatefulPartitionedCall
(tf_op_layer_ExpandDims_1/PartitionedCallPartitionedCallinput_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_tf_op_layer_ExpandDims_1_layer_call_and_return_conditional_losses_3988832*
(tf_op_layer_ExpandDims_1/PartitionedCallΝ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall1tf_op_layer_ExpandDims_1/PartitionedCall:output:0conv2d_3_399926conv2d_3_399928*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_3989022"
 conv2d_3/StatefulPartitionedCallΖ
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_9_399931batch_normalization_9_399933batch_normalization_9_399935batch_normalization_9_399937*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_3989552/
-batch_normalization_9/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0conv2d_4_399940conv2d_4_399942*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_3990022"
 conv2d_4/StatefulPartitionedCallΝ
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_10_399945batch_normalization_10_399947batch_normalization_10_399949batch_normalization_10_399951*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_39905520
.batch_normalization_10/StatefulPartitionedCallΤ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0conv2d_5_399954conv2d_5_399956*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_3991022"
 conv2d_5/StatefulPartitionedCallΞ
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_11_399959batch_normalization_11_399961batch_normalization_11_399963batch_normalization_11_399965*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_39915520
.batch_normalization_11/StatefulPartitionedCall·
%tf_op_layer_Reshape_1/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_3991972'
%tf_op_layer_Reshape_1/PartitionedCall
max_pooling1d_1/PartitionedCallPartitionedCall.tf_op_layer_Reshape_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_3980272!
max_pooling1d_1/PartitionedCall­
'tf_op_layer_Transpose_1/PartitionedCallPartitionedCall(max_pooling1d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_tf_op_layer_Transpose_1_layer_call_and_return_conditional_losses_3992122)
'tf_op_layer_Transpose_1/PartitionedCallΗ
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall0tf_op_layer_Transpose_1/PartitionedCall:output:0conv1d_2_399971conv1d_2_399973*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_3992362"
 conv1d_2/StatefulPartitionedCallΘ
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0batch_normalization_12_399976batch_normalization_12_399978batch_normalization_12_399980batch_normalization_12_399982*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:??????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_39930720
.batch_normalization_12/StatefulPartitionedCallΙ
dense_5/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0dense_5_399985dense_5_399987*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:??????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_3993742!
dense_5/StatefulPartitionedCallΗ
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0batch_normalization_13_399990batch_normalization_13_399992batch_normalization_13_399994batch_normalization_13_399996*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:??????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_39944520
.batch_normalization_13/StatefulPartitionedCallΞ
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0conv1d_3_399999conv1d_3_400001*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv1d_3_layer_call_and_return_conditional_losses_3994972"
 conv1d_3/StatefulPartitionedCallΘ
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0batch_normalization_14_400004batch_normalization_14_400006batch_normalization_14_400008batch_normalization_14_400010*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_39956820
.batch_normalization_14/StatefulPartitionedCallΙ
dense_6/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0dense_6_400013dense_6_400015*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_3996352!
dense_6/StatefulPartitionedCallΗ
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0batch_normalization_15_400018batch_normalization_15_400020batch_normalization_15_400022batch_normalization_15_400024*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_39970620
.batch_normalization_15/StatefulPartitionedCall
flatten_1/PartitionedCallPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_3997482
flatten_1/PartitionedCall±
dense_7/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_7_400028dense_7_400030*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_3997672!
dense_7/StatefulPartitionedCallΔ
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0batch_normalization_16_400033batch_normalization_16_400035batch_normalization_16_400037batch_normalization_16_400039*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_39872220
.batch_normalization_16/StatefulPartitionedCallΕ
dense_8/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0dense_8_400042dense_8_400044*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_3998292!
dense_8/StatefulPartitionedCallΓ
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0batch_normalization_17_400047batch_normalization_17_400049batch_normalization_17_400051batch_normalization_17_400053*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_39886220
.batch_normalization_17/StatefulPartitionedCallΕ
dense_9/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0dense_9_400056dense_9_400058*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_3998912!
dense_9/StatefulPartitionedCall
!tf_op_layer_Mul_1/PartitionedCallPartitionedCall(dense_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_tf_op_layer_Mul_1_layer_call_and_return_conditional_losses_3999132#
!tf_op_layer_Mul_1/PartitionedCall
IdentityIdentity*tf_op_layer_Mul_1/PartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*
_input_shapesϋ
ψ:?????????::::::::::::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:U Q
,
_output_shapes
:?????????
!
_user_specified_name	input_2
δ
Ρ
$__inference_signature_wrapper_400707
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54
identity’StatefulPartitionedCallΒ
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54*D
Tin=
;29*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*Z
_read_only_resource_inputs<
:8	
 !"#$%&'()*+,-./012345678*2
config_proto" 

CPU

GPU2 *0J 8 **
f%R#
!__inference__wrapped_model_3977062
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*
_input_shapesϋ
ψ:?????????::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:?????????
!
_user_specified_name	input_2
Ι
?
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_401710

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
Μ
ͺ
7__inference_batch_normalization_13_layer_call_fn_402452

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:??????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_3994252
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:??????????@2

Identity"
identityIdentity:output:0*:
_input_shapes)
':??????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:??????????@
 
_user_specified_nameinputs
ζ
ͺ
7__inference_batch_normalization_11_layer_call_fn_402050

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_3991552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:?????????
 
_user_specified_nameinputs

~
)__inference_conv2d_3_layer_call_fn_401626

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_3989022
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Β
ͺ
7__inference_batch_normalization_16_layer_call_fn_402971

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall’
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_3987222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs

?
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_398937

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ω
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:?????????@
 
_user_specified_nameinputs
	
¬
D__inference_conv2d_3_layer_call_and_return_conditional_losses_398902

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp₯
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????@2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
¦

R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_398007

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1α
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????:::::j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs


R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_399445

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:??????????@2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:??????????@2
batchnorm/add_1k
IdentityIdentitybatchnorm/add_1:z:0*
T0*+
_output_shapes
:??????????@2

Identity"
identityIdentity:output:0*:
_input_shapes)
':??????????@:::::S O
+
_output_shapes
:??????????@
 
_user_specified_nameinputs

?
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_401646

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ω
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:?????????@
 
_user_specified_nameinputs
¦

R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_401960

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1α
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????:::::j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
€
ͺ
7__inference_batch_normalization_10_layer_call_fn_401825

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCallΉ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_3978722
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
Υ)
Λ
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_402812

inputs
assignmovingavg_402787
assignmovingavg_1_402793)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity’#AssignMovingAvg/AssignSubVariableOp’%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradient¨
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:?????????@2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesΆ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze_1
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/402787*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_402787*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpΓ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/402787*
_output_shapes
:@2
AssignMovingAvg/subΊ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/402787*
_output_shapes
:@2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_402787AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/402787*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp€
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/402793*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_402793*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpΝ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/402793*
_output_shapes
:@2
AssignMovingAvg_1/subΔ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/402793*
_output_shapes
:@2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_402793AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/402793*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:?????????@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????@2
batchnorm/add_1Ή
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*+
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs


R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_399568

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2
batchnorm/add_1k
IdentityIdentitybatchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????:::::S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
Ι
i
M__inference_tf_op_layer_Mul_1_layer_call_and_return_conditional_losses_399913

inputs
identitys
Mul_1/yConst*
_output_shapes
:*
dtype0*-
value$B""  @@  @@  @@  @@  @@  @@2	
Mul_1/yp
Mul_1MulinputsMul_1/y:output:0*
T0*
_cloned(*'
_output_shapes
:?????????2
Mul_1]
IdentityIdentity	Mul_1:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs


R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_402235

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:??????????2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:??????????2
batchnorm/add_1k
IdentityIdentitybatchnorm/add_1:z:0*
T0*+
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':??????????:::::S O
+
_output_shapes
:??????????
 
_user_specified_nameinputs
?
ϊ
H__inference_functional_3_layer_call_and_return_conditional_losses_400207

inputs
conv2d_3_400070
conv2d_3_400072 
batch_normalization_9_400075 
batch_normalization_9_400077 
batch_normalization_9_400079 
batch_normalization_9_400081
conv2d_4_400084
conv2d_4_400086!
batch_normalization_10_400089!
batch_normalization_10_400091!
batch_normalization_10_400093!
batch_normalization_10_400095
conv2d_5_400098
conv2d_5_400100!
batch_normalization_11_400103!
batch_normalization_11_400105!
batch_normalization_11_400107!
batch_normalization_11_400109
conv1d_2_400115
conv1d_2_400117!
batch_normalization_12_400120!
batch_normalization_12_400122!
batch_normalization_12_400124!
batch_normalization_12_400126
dense_5_400129
dense_5_400131!
batch_normalization_13_400134!
batch_normalization_13_400136!
batch_normalization_13_400138!
batch_normalization_13_400140
conv1d_3_400143
conv1d_3_400145!
batch_normalization_14_400148!
batch_normalization_14_400150!
batch_normalization_14_400152!
batch_normalization_14_400154
dense_6_400157
dense_6_400159!
batch_normalization_15_400162!
batch_normalization_15_400164!
batch_normalization_15_400166!
batch_normalization_15_400168
dense_7_400172
dense_7_400174!
batch_normalization_16_400177!
batch_normalization_16_400179!
batch_normalization_16_400181!
batch_normalization_16_400183
dense_8_400186
dense_8_400188!
batch_normalization_17_400191!
batch_normalization_17_400193!
batch_normalization_17_400195!
batch_normalization_17_400197
dense_9_400200
dense_9_400202
identity’.batch_normalization_10/StatefulPartitionedCall’.batch_normalization_11/StatefulPartitionedCall’.batch_normalization_12/StatefulPartitionedCall’.batch_normalization_13/StatefulPartitionedCall’.batch_normalization_14/StatefulPartitionedCall’.batch_normalization_15/StatefulPartitionedCall’.batch_normalization_16/StatefulPartitionedCall’.batch_normalization_17/StatefulPartitionedCall’-batch_normalization_9/StatefulPartitionedCall’ conv1d_2/StatefulPartitionedCall’ conv1d_3/StatefulPartitionedCall’ conv2d_3/StatefulPartitionedCall’ conv2d_4/StatefulPartitionedCall’ conv2d_5/StatefulPartitionedCall’dense_5/StatefulPartitionedCall’dense_6/StatefulPartitionedCall’dense_7/StatefulPartitionedCall’dense_8/StatefulPartitionedCall’dense_9/StatefulPartitionedCall
(tf_op_layer_ExpandDims_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_tf_op_layer_ExpandDims_1_layer_call_and_return_conditional_losses_3988832*
(tf_op_layer_ExpandDims_1/PartitionedCallΝ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall1tf_op_layer_ExpandDims_1/PartitionedCall:output:0conv2d_3_400070conv2d_3_400072*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_3989022"
 conv2d_3/StatefulPartitionedCallΔ
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_9_400075batch_normalization_9_400077batch_normalization_9_400079batch_normalization_9_400081*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_3989372/
-batch_normalization_9/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0conv2d_4_400084conv2d_4_400086*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_3990022"
 conv2d_4/StatefulPartitionedCallΛ
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_10_400089batch_normalization_10_400091batch_normalization_10_400093batch_normalization_10_400095*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_39903720
.batch_normalization_10/StatefulPartitionedCallΤ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0conv2d_5_400098conv2d_5_400100*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_3991022"
 conv2d_5/StatefulPartitionedCallΜ
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_11_400103batch_normalization_11_400105batch_normalization_11_400107batch_normalization_11_400109*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_39913720
.batch_normalization_11/StatefulPartitionedCall·
%tf_op_layer_Reshape_1/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_3991972'
%tf_op_layer_Reshape_1/PartitionedCall
max_pooling1d_1/PartitionedCallPartitionedCall.tf_op_layer_Reshape_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_3980272!
max_pooling1d_1/PartitionedCall­
'tf_op_layer_Transpose_1/PartitionedCallPartitionedCall(max_pooling1d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_tf_op_layer_Transpose_1_layer_call_and_return_conditional_losses_3992122)
'tf_op_layer_Transpose_1/PartitionedCallΗ
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall0tf_op_layer_Transpose_1/PartitionedCall:output:0conv1d_2_400115conv1d_2_400117*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_3992362"
 conv1d_2/StatefulPartitionedCallΖ
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0batch_normalization_12_400120batch_normalization_12_400122batch_normalization_12_400124batch_normalization_12_400126*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_39928720
.batch_normalization_12/StatefulPartitionedCallΙ
dense_5/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0dense_5_400129dense_5_400131*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:??????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_3993742!
dense_5/StatefulPartitionedCallΕ
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0batch_normalization_13_400134batch_normalization_13_400136batch_normalization_13_400138batch_normalization_13_400140*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:??????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_39942520
.batch_normalization_13/StatefulPartitionedCallΞ
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0conv1d_3_400143conv1d_3_400145*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv1d_3_layer_call_and_return_conditional_losses_3994972"
 conv1d_3/StatefulPartitionedCallΖ
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0batch_normalization_14_400148batch_normalization_14_400150batch_normalization_14_400152batch_normalization_14_400154*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_39954820
.batch_normalization_14/StatefulPartitionedCallΙ
dense_6/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0dense_6_400157dense_6_400159*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_3996352!
dense_6/StatefulPartitionedCallΕ
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0batch_normalization_15_400162batch_normalization_15_400164batch_normalization_15_400166batch_normalization_15_400168*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_39968620
.batch_normalization_15/StatefulPartitionedCall
flatten_1/PartitionedCallPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_3997482
flatten_1/PartitionedCall±
dense_7/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_7_400172dense_7_400174*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_3997672!
dense_7/StatefulPartitionedCallΒ
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0batch_normalization_16_400177batch_normalization_16_400179batch_normalization_16_400181batch_normalization_16_400183*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_39868920
.batch_normalization_16/StatefulPartitionedCallΕ
dense_8/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0dense_8_400186dense_8_400188*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_3998292!
dense_8/StatefulPartitionedCallΑ
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0batch_normalization_17_400191batch_normalization_17_400193batch_normalization_17_400195batch_normalization_17_400197*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_39882920
.batch_normalization_17/StatefulPartitionedCallΕ
dense_9/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0dense_9_400200dense_9_400202*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_3998912!
dense_9/StatefulPartitionedCall
!tf_op_layer_Mul_1/PartitionedCallPartitionedCall(dense_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_tf_op_layer_Mul_1_layer_call_and_return_conditional_losses_3999132#
!tf_op_layer_Mul_1/PartitionedCall
IdentityIdentity*tf_op_layer_Mul_1/PartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*
_input_shapesϋ
ψ:?????????::::::::::::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs


R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_398722

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:?????????2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:?????????2
batchnorm/add_1h
IdentityIdentitybatchnorm/add_1:z:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::::P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
Ζ

R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_398162

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1t
IdentityIdentitybatchnorm/add_1:z:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????:::::\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
α
}
(__inference_dense_8_layer_call_fn_402991

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallψ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_3998292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
ο
}
(__inference_dense_5_layer_call_fn_402301

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallό
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:??????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_3993742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:??????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:??????????
 
_user_specified_nameinputs
κ
g
K__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_398027

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims³
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize	
*
paddingVALID*
strides	
2	
MaxPool
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
±
«
C__inference_dense_7_layer_call_and_return_conditional_losses_402880

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:::P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs

Ή
D__inference_conv1d_2_layer_call_and_return_conditional_losses_399236

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????2
conv1d/ExpandDimsΈ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1·
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:??????????*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:??????????*
squeeze_dims

ύ????????2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:??????????2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:??????????2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :?????????:::T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
Μ
ͺ
7__inference_batch_normalization_14_layer_call_fn_402641

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_3995482
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs

m
Q__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_399197

inputs
identityw
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_1/shape
	Reshape_1ReshapeinputsReshape_1/shape:output:0*
T0*
_cloned(*-
_output_shapes
:?????????2
	Reshape_1l
IdentityIdentityReshape_1:output:0*
T0*-
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:?????????:Y U
1
_output_shapes
:?????????
 
_user_specified_nameinputs
₯
§@
__inference__traced_save_403544
file_prefix.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop:
6savev2_batch_normalization_9_gamma_read_readvariableop9
5savev2_batch_normalization_9_beta_read_readvariableop@
<savev2_batch_normalization_9_moving_mean_read_readvariableopD
@savev2_batch_normalization_9_moving_variance_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop;
7savev2_batch_normalization_10_gamma_read_readvariableop:
6savev2_batch_normalization_10_beta_read_readvariableopA
=savev2_batch_normalization_10_moving_mean_read_readvariableopE
Asavev2_batch_normalization_10_moving_variance_read_readvariableop.
*savev2_conv2d_5_kernel_read_readvariableop,
(savev2_conv2d_5_bias_read_readvariableop;
7savev2_batch_normalization_11_gamma_read_readvariableop:
6savev2_batch_normalization_11_beta_read_readvariableopA
=savev2_batch_normalization_11_moving_mean_read_readvariableopE
Asavev2_batch_normalization_11_moving_variance_read_readvariableop.
*savev2_conv1d_2_kernel_read_readvariableop,
(savev2_conv1d_2_bias_read_readvariableop;
7savev2_batch_normalization_12_gamma_read_readvariableop:
6savev2_batch_normalization_12_beta_read_readvariableopA
=savev2_batch_normalization_12_moving_mean_read_readvariableopE
Asavev2_batch_normalization_12_moving_variance_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop;
7savev2_batch_normalization_13_gamma_read_readvariableop:
6savev2_batch_normalization_13_beta_read_readvariableopA
=savev2_batch_normalization_13_moving_mean_read_readvariableopE
Asavev2_batch_normalization_13_moving_variance_read_readvariableop.
*savev2_conv1d_3_kernel_read_readvariableop,
(savev2_conv1d_3_bias_read_readvariableop;
7savev2_batch_normalization_14_gamma_read_readvariableop:
6savev2_batch_normalization_14_beta_read_readvariableopA
=savev2_batch_normalization_14_moving_mean_read_readvariableopE
Asavev2_batch_normalization_14_moving_variance_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop;
7savev2_batch_normalization_15_gamma_read_readvariableop:
6savev2_batch_normalization_15_beta_read_readvariableopA
=savev2_batch_normalization_15_moving_mean_read_readvariableopE
Asavev2_batch_normalization_15_moving_variance_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop;
7savev2_batch_normalization_16_gamma_read_readvariableop:
6savev2_batch_normalization_16_beta_read_readvariableopA
=savev2_batch_normalization_16_moving_mean_read_readvariableopE
Asavev2_batch_normalization_16_moving_variance_read_readvariableop-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop;
7savev2_batch_normalization_17_gamma_read_readvariableop:
6savev2_batch_normalization_17_beta_read_readvariableopA
=savev2_batch_normalization_17_moving_mean_read_readvariableopE
Asavev2_batch_normalization_17_moving_variance_read_readvariableop-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_conv2d_3_kernel_m_read_readvariableop3
/savev2_adam_conv2d_3_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_9_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_9_beta_m_read_readvariableop5
1savev2_adam_conv2d_4_kernel_m_read_readvariableop3
/savev2_adam_conv2d_4_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_10_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_10_beta_m_read_readvariableop5
1savev2_adam_conv2d_5_kernel_m_read_readvariableop3
/savev2_adam_conv2d_5_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_11_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_11_beta_m_read_readvariableop5
1savev2_adam_conv1d_2_kernel_m_read_readvariableop3
/savev2_adam_conv1d_2_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_12_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_12_beta_m_read_readvariableop4
0savev2_adam_dense_5_kernel_m_read_readvariableop2
.savev2_adam_dense_5_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_13_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_13_beta_m_read_readvariableop5
1savev2_adam_conv1d_3_kernel_m_read_readvariableop3
/savev2_adam_conv1d_3_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_14_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_14_beta_m_read_readvariableop4
0savev2_adam_dense_6_kernel_m_read_readvariableop2
.savev2_adam_dense_6_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_15_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_15_beta_m_read_readvariableop4
0savev2_adam_dense_7_kernel_m_read_readvariableop2
.savev2_adam_dense_7_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_16_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_16_beta_m_read_readvariableop4
0savev2_adam_dense_8_kernel_m_read_readvariableop2
.savev2_adam_dense_8_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_17_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_17_beta_m_read_readvariableop4
0savev2_adam_dense_9_kernel_m_read_readvariableop2
.savev2_adam_dense_9_bias_m_read_readvariableop5
1savev2_adam_conv2d_3_kernel_v_read_readvariableop3
/savev2_adam_conv2d_3_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_9_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_9_beta_v_read_readvariableop5
1savev2_adam_conv2d_4_kernel_v_read_readvariableop3
/savev2_adam_conv2d_4_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_10_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_10_beta_v_read_readvariableop5
1savev2_adam_conv2d_5_kernel_v_read_readvariableop3
/savev2_adam_conv2d_5_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_11_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_11_beta_v_read_readvariableop5
1savev2_adam_conv1d_2_kernel_v_read_readvariableop3
/savev2_adam_conv1d_2_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_12_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_12_beta_v_read_readvariableop4
0savev2_adam_dense_5_kernel_v_read_readvariableop2
.savev2_adam_dense_5_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_13_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_13_beta_v_read_readvariableop5
1savev2_adam_conv1d_3_kernel_v_read_readvariableop3
/savev2_adam_conv1d_3_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_14_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_14_beta_v_read_readvariableop4
0savev2_adam_dense_6_kernel_v_read_readvariableop2
.savev2_adam_dense_6_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_15_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_15_beta_v_read_readvariableop4
0savev2_adam_dense_7_kernel_v_read_readvariableop2
.savev2_adam_dense_7_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_16_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_16_beta_v_read_readvariableop4
0savev2_adam_dense_8_kernel_v_read_readvariableop2
.savev2_adam_dense_8_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_17_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_17_beta_v_read_readvariableop4
0savev2_adam_dense_9_kernel_v_read_readvariableop2
.savev2_adam_dense_9_bias_v_read_readvariableop
savev2_const

identity_1’MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_d2d6e7e0c0934c4e8b9265157873b62a/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameΣN
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*δM
valueΪMBΧMB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-17/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-17/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names₯
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*?
value€B‘B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesΨ=
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop6savev2_batch_normalization_9_gamma_read_readvariableop5savev2_batch_normalization_9_beta_read_readvariableop<savev2_batch_normalization_9_moving_mean_read_readvariableop@savev2_batch_normalization_9_moving_variance_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop7savev2_batch_normalization_10_gamma_read_readvariableop6savev2_batch_normalization_10_beta_read_readvariableop=savev2_batch_normalization_10_moving_mean_read_readvariableopAsavev2_batch_normalization_10_moving_variance_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop7savev2_batch_normalization_11_gamma_read_readvariableop6savev2_batch_normalization_11_beta_read_readvariableop=savev2_batch_normalization_11_moving_mean_read_readvariableopAsavev2_batch_normalization_11_moving_variance_read_readvariableop*savev2_conv1d_2_kernel_read_readvariableop(savev2_conv1d_2_bias_read_readvariableop7savev2_batch_normalization_12_gamma_read_readvariableop6savev2_batch_normalization_12_beta_read_readvariableop=savev2_batch_normalization_12_moving_mean_read_readvariableopAsavev2_batch_normalization_12_moving_variance_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop7savev2_batch_normalization_13_gamma_read_readvariableop6savev2_batch_normalization_13_beta_read_readvariableop=savev2_batch_normalization_13_moving_mean_read_readvariableopAsavev2_batch_normalization_13_moving_variance_read_readvariableop*savev2_conv1d_3_kernel_read_readvariableop(savev2_conv1d_3_bias_read_readvariableop7savev2_batch_normalization_14_gamma_read_readvariableop6savev2_batch_normalization_14_beta_read_readvariableop=savev2_batch_normalization_14_moving_mean_read_readvariableopAsavev2_batch_normalization_14_moving_variance_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop7savev2_batch_normalization_15_gamma_read_readvariableop6savev2_batch_normalization_15_beta_read_readvariableop=savev2_batch_normalization_15_moving_mean_read_readvariableopAsavev2_batch_normalization_15_moving_variance_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop7savev2_batch_normalization_16_gamma_read_readvariableop6savev2_batch_normalization_16_beta_read_readvariableop=savev2_batch_normalization_16_moving_mean_read_readvariableopAsavev2_batch_normalization_16_moving_variance_read_readvariableop)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop7savev2_batch_normalization_17_gamma_read_readvariableop6savev2_batch_normalization_17_beta_read_readvariableop=savev2_batch_normalization_17_moving_mean_read_readvariableopAsavev2_batch_normalization_17_moving_variance_read_readvariableop)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_conv2d_3_kernel_m_read_readvariableop/savev2_adam_conv2d_3_bias_m_read_readvariableop=savev2_adam_batch_normalization_9_gamma_m_read_readvariableop<savev2_adam_batch_normalization_9_beta_m_read_readvariableop1savev2_adam_conv2d_4_kernel_m_read_readvariableop/savev2_adam_conv2d_4_bias_m_read_readvariableop>savev2_adam_batch_normalization_10_gamma_m_read_readvariableop=savev2_adam_batch_normalization_10_beta_m_read_readvariableop1savev2_adam_conv2d_5_kernel_m_read_readvariableop/savev2_adam_conv2d_5_bias_m_read_readvariableop>savev2_adam_batch_normalization_11_gamma_m_read_readvariableop=savev2_adam_batch_normalization_11_beta_m_read_readvariableop1savev2_adam_conv1d_2_kernel_m_read_readvariableop/savev2_adam_conv1d_2_bias_m_read_readvariableop>savev2_adam_batch_normalization_12_gamma_m_read_readvariableop=savev2_adam_batch_normalization_12_beta_m_read_readvariableop0savev2_adam_dense_5_kernel_m_read_readvariableop.savev2_adam_dense_5_bias_m_read_readvariableop>savev2_adam_batch_normalization_13_gamma_m_read_readvariableop=savev2_adam_batch_normalization_13_beta_m_read_readvariableop1savev2_adam_conv1d_3_kernel_m_read_readvariableop/savev2_adam_conv1d_3_bias_m_read_readvariableop>savev2_adam_batch_normalization_14_gamma_m_read_readvariableop=savev2_adam_batch_normalization_14_beta_m_read_readvariableop0savev2_adam_dense_6_kernel_m_read_readvariableop.savev2_adam_dense_6_bias_m_read_readvariableop>savev2_adam_batch_normalization_15_gamma_m_read_readvariableop=savev2_adam_batch_normalization_15_beta_m_read_readvariableop0savev2_adam_dense_7_kernel_m_read_readvariableop.savev2_adam_dense_7_bias_m_read_readvariableop>savev2_adam_batch_normalization_16_gamma_m_read_readvariableop=savev2_adam_batch_normalization_16_beta_m_read_readvariableop0savev2_adam_dense_8_kernel_m_read_readvariableop.savev2_adam_dense_8_bias_m_read_readvariableop>savev2_adam_batch_normalization_17_gamma_m_read_readvariableop=savev2_adam_batch_normalization_17_beta_m_read_readvariableop0savev2_adam_dense_9_kernel_m_read_readvariableop.savev2_adam_dense_9_bias_m_read_readvariableop1savev2_adam_conv2d_3_kernel_v_read_readvariableop/savev2_adam_conv2d_3_bias_v_read_readvariableop=savev2_adam_batch_normalization_9_gamma_v_read_readvariableop<savev2_adam_batch_normalization_9_beta_v_read_readvariableop1savev2_adam_conv2d_4_kernel_v_read_readvariableop/savev2_adam_conv2d_4_bias_v_read_readvariableop>savev2_adam_batch_normalization_10_gamma_v_read_readvariableop=savev2_adam_batch_normalization_10_beta_v_read_readvariableop1savev2_adam_conv2d_5_kernel_v_read_readvariableop/savev2_adam_conv2d_5_bias_v_read_readvariableop>savev2_adam_batch_normalization_11_gamma_v_read_readvariableop=savev2_adam_batch_normalization_11_beta_v_read_readvariableop1savev2_adam_conv1d_2_kernel_v_read_readvariableop/savev2_adam_conv1d_2_bias_v_read_readvariableop>savev2_adam_batch_normalization_12_gamma_v_read_readvariableop=savev2_adam_batch_normalization_12_beta_v_read_readvariableop0savev2_adam_dense_5_kernel_v_read_readvariableop.savev2_adam_dense_5_bias_v_read_readvariableop>savev2_adam_batch_normalization_13_gamma_v_read_readvariableop=savev2_adam_batch_normalization_13_beta_v_read_readvariableop1savev2_adam_conv1d_3_kernel_v_read_readvariableop/savev2_adam_conv1d_3_bias_v_read_readvariableop>savev2_adam_batch_normalization_14_gamma_v_read_readvariableop=savev2_adam_batch_normalization_14_beta_v_read_readvariableop0savev2_adam_dense_6_kernel_v_read_readvariableop.savev2_adam_dense_6_bias_v_read_readvariableop>savev2_adam_batch_normalization_15_gamma_v_read_readvariableop=savev2_adam_batch_normalization_15_beta_v_read_readvariableop0savev2_adam_dense_7_kernel_v_read_readvariableop.savev2_adam_dense_7_bias_v_read_readvariableop>savev2_adam_batch_normalization_16_gamma_v_read_readvariableop=savev2_adam_batch_normalization_16_beta_v_read_readvariableop0savev2_adam_dense_8_kernel_v_read_readvariableop.savev2_adam_dense_8_bias_v_read_readvariableop>savev2_adam_batch_normalization_17_gamma_v_read_readvariableop=savev2_adam_batch_normalization_17_beta_v_read_readvariableop0savev2_adam_dense_9_kernel_v_read_readvariableop.savev2_adam_dense_9_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2Ί
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes‘
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Ή
_input_shapes§
€: :@:@:@:@:@:@:@@:@:@:@:@:@:@::::::::::::@:@:@:@:@:@:@::::::@:@:@:@:@:@:
::::::	@:@:@:@:@:@:@:: : : : : : : :@:@:@:@:@@:@:@:@:@::::::::@:@:@:@:@::::@:@:@:@:
::::	@:@:@:@:@::@:@:@:@:@@:@:@:@:@::::::::@:@:@:@:@::::@:@:@:@:
::::	@:@:@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 	

_output_shapes
:@: 


_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@:!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::($
"
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:($
"
_output_shapes
:@:  

_output_shapes
:: !

_output_shapes
:: "

_output_shapes
:: #

_output_shapes
:: $

_output_shapes
::$% 

_output_shapes

:@: &

_output_shapes
:@: '

_output_shapes
:@: (

_output_shapes
:@: )

_output_shapes
:@: *

_output_shapes
:@:&+"
 
_output_shapes
:
:!,

_output_shapes	
::!-

_output_shapes	
::!.

_output_shapes	
::!/

_output_shapes	
::!0

_output_shapes	
::%1!

_output_shapes
:	@: 2

_output_shapes
:@: 3

_output_shapes
:@: 4

_output_shapes
:@: 5

_output_shapes
:@: 6

_output_shapes
:@:$7 

_output_shapes

:@: 8

_output_shapes
::9

_output_shapes
: ::

_output_shapes
: :;

_output_shapes
: :<

_output_shapes
: :=

_output_shapes
: :>

_output_shapes
: :?

_output_shapes
: :,@(
&
_output_shapes
:@: A

_output_shapes
:@: B

_output_shapes
:@: C

_output_shapes
:@:,D(
&
_output_shapes
:@@: E

_output_shapes
:@: F

_output_shapes
:@: G

_output_shapes
:@:-H)
'
_output_shapes
:@:!I

_output_shapes	
::!J

_output_shapes	
::!K

_output_shapes	
::(L$
"
_output_shapes
:: M

_output_shapes
:: N

_output_shapes
:: O

_output_shapes
::$P 

_output_shapes

:@: Q

_output_shapes
:@: R

_output_shapes
:@: S

_output_shapes
:@:(T$
"
_output_shapes
:@: U

_output_shapes
:: V

_output_shapes
:: W

_output_shapes
::$X 

_output_shapes

:@: Y

_output_shapes
:@: Z

_output_shapes
:@: [

_output_shapes
:@:&\"
 
_output_shapes
:
:!]

_output_shapes	
::!^

_output_shapes	
::!_

_output_shapes	
::%`!

_output_shapes
:	@: a

_output_shapes
:@: b

_output_shapes
:@: c

_output_shapes
:@:$d 

_output_shapes

:@: e

_output_shapes
::,f(
&
_output_shapes
:@: g

_output_shapes
:@: h

_output_shapes
:@: i

_output_shapes
:@:,j(
&
_output_shapes
:@@: k

_output_shapes
:@: l

_output_shapes
:@: m

_output_shapes
:@:-n)
'
_output_shapes
:@:!o

_output_shapes	
::!p

_output_shapes	
::!q

_output_shapes	
::(r$
"
_output_shapes
:: s

_output_shapes
:: t

_output_shapes
:: u

_output_shapes
::$v 

_output_shapes

:@: w

_output_shapes
:@: x

_output_shapes
:@: y

_output_shapes
:@:(z$
"
_output_shapes
:@: {

_output_shapes
:: |

_output_shapes
:: }

_output_shapes
::$~ 

_output_shapes

:@: 

_output_shapes
:@:!

_output_shapes
:@:!

_output_shapes
:@:'"
 
_output_shapes
:
:"

_output_shapes	
::"

_output_shapes	
::"

_output_shapes	
::&!

_output_shapes
:	@:!

_output_shapes
:@:!

_output_shapes
:@:!

_output_shapes
:@:% 

_output_shapes

:@:!

_output_shapes
::

_output_shapes
: 
Φ
―
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_401942

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ο
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
δ
ͺ
7__inference_batch_normalization_11_layer_call_fn_402037

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_3991372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:?????????
 
_user_specified_nameinputs
¨
ͺ
7__inference_batch_normalization_11_layer_call_fn_401973

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCallΊ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_3979762
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs

Ω
-__inference_functional_3_layer_call_fn_401595

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54
identity’StatefulPartitionedCallθ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54*D
Tin=
;29*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*Z
_read_only_resource_inputs<
:8	
 !"#$%&'()*+,-./012345678*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_functional_3_layer_call_and_return_conditional_losses_4004652
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*
_input_shapesϋ
ψ:?????????::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
Κ

!__inference__wrapped_model_397706
input_28
4functional_3_conv2d_3_conv2d_readvariableop_resource9
5functional_3_conv2d_3_biasadd_readvariableop_resource>
:functional_3_batch_normalization_9_readvariableop_resource@
<functional_3_batch_normalization_9_readvariableop_1_resourceO
Kfunctional_3_batch_normalization_9_fusedbatchnormv3_readvariableop_resourceQ
Mfunctional_3_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource8
4functional_3_conv2d_4_conv2d_readvariableop_resource9
5functional_3_conv2d_4_biasadd_readvariableop_resource?
;functional_3_batch_normalization_10_readvariableop_resourceA
=functional_3_batch_normalization_10_readvariableop_1_resourceP
Lfunctional_3_batch_normalization_10_fusedbatchnormv3_readvariableop_resourceR
Nfunctional_3_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource8
4functional_3_conv2d_5_conv2d_readvariableop_resource9
5functional_3_conv2d_5_biasadd_readvariableop_resource?
;functional_3_batch_normalization_11_readvariableop_resourceA
=functional_3_batch_normalization_11_readvariableop_1_resourceP
Lfunctional_3_batch_normalization_11_fusedbatchnormv3_readvariableop_resourceR
Nfunctional_3_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resourceE
Afunctional_3_conv1d_2_conv1d_expanddims_1_readvariableop_resource9
5functional_3_conv1d_2_biasadd_readvariableop_resourceI
Efunctional_3_batch_normalization_12_batchnorm_readvariableop_resourceM
Ifunctional_3_batch_normalization_12_batchnorm_mul_readvariableop_resourceK
Gfunctional_3_batch_normalization_12_batchnorm_readvariableop_1_resourceK
Gfunctional_3_batch_normalization_12_batchnorm_readvariableop_2_resource:
6functional_3_dense_5_tensordot_readvariableop_resource8
4functional_3_dense_5_biasadd_readvariableop_resourceI
Efunctional_3_batch_normalization_13_batchnorm_readvariableop_resourceM
Ifunctional_3_batch_normalization_13_batchnorm_mul_readvariableop_resourceK
Gfunctional_3_batch_normalization_13_batchnorm_readvariableop_1_resourceK
Gfunctional_3_batch_normalization_13_batchnorm_readvariableop_2_resourceE
Afunctional_3_conv1d_3_conv1d_expanddims_1_readvariableop_resource9
5functional_3_conv1d_3_biasadd_readvariableop_resourceI
Efunctional_3_batch_normalization_14_batchnorm_readvariableop_resourceM
Ifunctional_3_batch_normalization_14_batchnorm_mul_readvariableop_resourceK
Gfunctional_3_batch_normalization_14_batchnorm_readvariableop_1_resourceK
Gfunctional_3_batch_normalization_14_batchnorm_readvariableop_2_resource:
6functional_3_dense_6_tensordot_readvariableop_resource8
4functional_3_dense_6_biasadd_readvariableop_resourceI
Efunctional_3_batch_normalization_15_batchnorm_readvariableop_resourceM
Ifunctional_3_batch_normalization_15_batchnorm_mul_readvariableop_resourceK
Gfunctional_3_batch_normalization_15_batchnorm_readvariableop_1_resourceK
Gfunctional_3_batch_normalization_15_batchnorm_readvariableop_2_resource7
3functional_3_dense_7_matmul_readvariableop_resource8
4functional_3_dense_7_biasadd_readvariableop_resourceI
Efunctional_3_batch_normalization_16_batchnorm_readvariableop_resourceM
Ifunctional_3_batch_normalization_16_batchnorm_mul_readvariableop_resourceK
Gfunctional_3_batch_normalization_16_batchnorm_readvariableop_1_resourceK
Gfunctional_3_batch_normalization_16_batchnorm_readvariableop_2_resource7
3functional_3_dense_8_matmul_readvariableop_resource8
4functional_3_dense_8_biasadd_readvariableop_resourceI
Efunctional_3_batch_normalization_17_batchnorm_readvariableop_resourceM
Ifunctional_3_batch_normalization_17_batchnorm_mul_readvariableop_resourceK
Gfunctional_3_batch_normalization_17_batchnorm_readvariableop_1_resourceK
Gfunctional_3_batch_normalization_17_batchnorm_readvariableop_2_resource7
3functional_3_dense_9_matmul_readvariableop_resource8
4functional_3_dense_9_biasadd_readvariableop_resource
identity»
6functional_3/tf_op_layer_ExpandDims_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????28
6functional_3/tf_op_layer_ExpandDims_1/ExpandDims_1/dim
2functional_3/tf_op_layer_ExpandDims_1/ExpandDims_1
ExpandDimsinput_2?functional_3/tf_op_layer_ExpandDims_1/ExpandDims_1/dim:output:0*
T0*
_cloned(*0
_output_shapes
:?????????24
2functional_3/tf_op_layer_ExpandDims_1/ExpandDims_1Χ
+functional_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOp4functional_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02-
+functional_3/conv2d_3/Conv2D/ReadVariableOp
functional_3/conv2d_3/Conv2DConv2D;functional_3/tf_op_layer_ExpandDims_1/ExpandDims_1:output:03functional_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@*
paddingVALID*
strides
2
functional_3/conv2d_3/Conv2DΞ
,functional_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp5functional_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,functional_3/conv2d_3/BiasAdd/ReadVariableOpα
functional_3/conv2d_3/BiasAddBiasAdd%functional_3/conv2d_3/Conv2D:output:04functional_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@2
functional_3/conv2d_3/BiasAdd£
functional_3/conv2d_3/ReluRelu&functional_3/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@2
functional_3/conv2d_3/Reluέ
1functional_3/batch_normalization_9/ReadVariableOpReadVariableOp:functional_3_batch_normalization_9_readvariableop_resource*
_output_shapes
:@*
dtype023
1functional_3/batch_normalization_9/ReadVariableOpγ
3functional_3/batch_normalization_9/ReadVariableOp_1ReadVariableOp<functional_3_batch_normalization_9_readvariableop_1_resource*
_output_shapes
:@*
dtype025
3functional_3/batch_normalization_9/ReadVariableOp_1
Bfunctional_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpKfunctional_3_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02D
Bfunctional_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOp
Dfunctional_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMfunctional_3_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02F
Dfunctional_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Ώ
3functional_3/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3(functional_3/conv2d_3/Relu:activations:09functional_3/batch_normalization_9/ReadVariableOp:value:0;functional_3/batch_normalization_9/ReadVariableOp_1:value:0Jfunctional_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????@:@:@:@:@:*
epsilon%o:*
is_training( 25
3functional_3/batch_normalization_9/FusedBatchNormV3Χ
+functional_3/conv2d_4/Conv2D/ReadVariableOpReadVariableOp4functional_3_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02-
+functional_3/conv2d_4/Conv2D/ReadVariableOp
functional_3/conv2d_4/Conv2DConv2D7functional_3/batch_normalization_9/FusedBatchNormV3:y:03functional_3/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@*
paddingVALID*
strides
2
functional_3/conv2d_4/Conv2DΞ
,functional_3/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp5functional_3_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,functional_3/conv2d_4/BiasAdd/ReadVariableOpα
functional_3/conv2d_4/BiasAddBiasAdd%functional_3/conv2d_4/Conv2D:output:04functional_3/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@2
functional_3/conv2d_4/BiasAdd£
functional_3/conv2d_4/ReluRelu&functional_3/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@2
functional_3/conv2d_4/Reluΰ
2functional_3/batch_normalization_10/ReadVariableOpReadVariableOp;functional_3_batch_normalization_10_readvariableop_resource*
_output_shapes
:@*
dtype024
2functional_3/batch_normalization_10/ReadVariableOpζ
4functional_3/batch_normalization_10/ReadVariableOp_1ReadVariableOp=functional_3_batch_normalization_10_readvariableop_1_resource*
_output_shapes
:@*
dtype026
4functional_3/batch_normalization_10/ReadVariableOp_1
Cfunctional_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpLfunctional_3_batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02E
Cfunctional_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp
Efunctional_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNfunctional_3_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02G
Efunctional_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1Ε
4functional_3/batch_normalization_10/FusedBatchNormV3FusedBatchNormV3(functional_3/conv2d_4/Relu:activations:0:functional_3/batch_normalization_10/ReadVariableOp:value:0<functional_3/batch_normalization_10/ReadVariableOp_1:value:0Kfunctional_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0Mfunctional_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????@:@:@:@:@:*
epsilon%o:*
is_training( 26
4functional_3/batch_normalization_10/FusedBatchNormV3Ψ
+functional_3/conv2d_5/Conv2D/ReadVariableOpReadVariableOp4functional_3_conv2d_5_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02-
+functional_3/conv2d_5/Conv2D/ReadVariableOp
functional_3/conv2d_5/Conv2DConv2D8functional_3/batch_normalization_10/FusedBatchNormV3:y:03functional_3/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
2
functional_3/conv2d_5/Conv2DΟ
,functional_3/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp5functional_3_conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,functional_3/conv2d_5/BiasAdd/ReadVariableOpβ
functional_3/conv2d_5/BiasAddBiasAdd%functional_3/conv2d_5/Conv2D:output:04functional_3/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????2
functional_3/conv2d_5/BiasAdd€
functional_3/conv2d_5/ReluRelu&functional_3/conv2d_5/BiasAdd:output:0*
T0*1
_output_shapes
:?????????2
functional_3/conv2d_5/Reluα
2functional_3/batch_normalization_11/ReadVariableOpReadVariableOp;functional_3_batch_normalization_11_readvariableop_resource*
_output_shapes	
:*
dtype024
2functional_3/batch_normalization_11/ReadVariableOpη
4functional_3/batch_normalization_11/ReadVariableOp_1ReadVariableOp=functional_3_batch_normalization_11_readvariableop_1_resource*
_output_shapes	
:*
dtype026
4functional_3/batch_normalization_11/ReadVariableOp_1
Cfunctional_3/batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpLfunctional_3_batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02E
Cfunctional_3/batch_normalization_11/FusedBatchNormV3/ReadVariableOp
Efunctional_3/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNfunctional_3_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02G
Efunctional_3/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1Κ
4functional_3/batch_normalization_11/FusedBatchNormV3FusedBatchNormV3(functional_3/conv2d_5/Relu:activations:0:functional_3/batch_normalization_11/ReadVariableOp:value:0<functional_3/batch_normalization_11/ReadVariableOp_1:value:0Kfunctional_3/batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0Mfunctional_3/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:?????????:::::*
epsilon%o:*
is_training( 26
4functional_3/batch_normalization_11/FusedBatchNormV3½
2functional_3/tf_op_layer_Reshape_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      24
2functional_3/tf_op_layer_Reshape_1/Reshape_1/shape₯
,functional_3/tf_op_layer_Reshape_1/Reshape_1Reshape8functional_3/batch_normalization_11/FusedBatchNormV3:y:0;functional_3/tf_op_layer_Reshape_1/Reshape_1/shape:output:0*
T0*
_cloned(*-
_output_shapes
:?????????2.
,functional_3/tf_op_layer_Reshape_1/Reshape_1
+functional_3/max_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+functional_3/max_pooling1d_1/ExpandDims/dim
'functional_3/max_pooling1d_1/ExpandDims
ExpandDims5functional_3/tf_op_layer_Reshape_1/Reshape_1:output:04functional_3/max_pooling1d_1/ExpandDims/dim:output:0*
T0*1
_output_shapes
:?????????2)
'functional_3/max_pooling1d_1/ExpandDimsω
$functional_3/max_pooling1d_1/MaxPoolMaxPool0functional_3/max_pooling1d_1/ExpandDims:output:0*0
_output_shapes
:?????????*
ksize	
*
paddingVALID*
strides	
2&
$functional_3/max_pooling1d_1/MaxPoolΤ
$functional_3/max_pooling1d_1/SqueezeSqueeze-functional_3/max_pooling1d_1/MaxPool:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims
2&
$functional_3/max_pooling1d_1/SqueezeΓ
5functional_3/tf_op_layer_Transpose_1/Transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          27
5functional_3/tf_op_layer_Transpose_1/Transpose_1/perm¦
0functional_3/tf_op_layer_Transpose_1/Transpose_1	Transpose-functional_3/max_pooling1d_1/Squeeze:output:0>functional_3/tf_op_layer_Transpose_1/Transpose_1/perm:output:0*
T0*
_cloned(*,
_output_shapes
:?????????22
0functional_3/tf_op_layer_Transpose_1/Transpose_1₯
+functional_3/conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????2-
+functional_3/conv1d_2/conv1d/ExpandDims/dim
'functional_3/conv1d_2/conv1d/ExpandDims
ExpandDims4functional_3/tf_op_layer_Transpose_1/Transpose_1:y:04functional_3/conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????2)
'functional_3/conv1d_2/conv1d/ExpandDimsϊ
8functional_3/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpAfunctional_3_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02:
8functional_3/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp 
-functional_3/conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-functional_3/conv1d_2/conv1d/ExpandDims_1/dim
)functional_3/conv1d_2/conv1d/ExpandDims_1
ExpandDims@functional_3/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:06functional_3/conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2+
)functional_3/conv1d_2/conv1d/ExpandDims_1
functional_3/conv1d_2/conv1dConv2D0functional_3/conv1d_2/conv1d/ExpandDims:output:02functional_3/conv1d_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:??????????*
paddingVALID*
strides
2
functional_3/conv1d_2/conv1dΤ
$functional_3/conv1d_2/conv1d/SqueezeSqueeze%functional_3/conv1d_2/conv1d:output:0*
T0*+
_output_shapes
:??????????*
squeeze_dims

ύ????????2&
$functional_3/conv1d_2/conv1d/SqueezeΞ
,functional_3/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp5functional_3_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,functional_3/conv1d_2/BiasAdd/ReadVariableOpδ
functional_3/conv1d_2/BiasAddBiasAdd-functional_3/conv1d_2/conv1d/Squeeze:output:04functional_3/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:??????????2
functional_3/conv1d_2/BiasAdd
functional_3/conv1d_2/ReluRelu&functional_3/conv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:??????????2
functional_3/conv1d_2/Reluώ
<functional_3/batch_normalization_12/batchnorm/ReadVariableOpReadVariableOpEfunctional_3_batch_normalization_12_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02>
<functional_3/batch_normalization_12/batchnorm/ReadVariableOp―
3functional_3/batch_normalization_12/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:25
3functional_3/batch_normalization_12/batchnorm/add/y
1functional_3/batch_normalization_12/batchnorm/addAddV2Dfunctional_3/batch_normalization_12/batchnorm/ReadVariableOp:value:0<functional_3/batch_normalization_12/batchnorm/add/y:output:0*
T0*
_output_shapes
:23
1functional_3/batch_normalization_12/batchnorm/addΟ
3functional_3/batch_normalization_12/batchnorm/RsqrtRsqrt5functional_3/batch_normalization_12/batchnorm/add:z:0*
T0*
_output_shapes
:25
3functional_3/batch_normalization_12/batchnorm/Rsqrt
@functional_3/batch_normalization_12/batchnorm/mul/ReadVariableOpReadVariableOpIfunctional_3_batch_normalization_12_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02B
@functional_3/batch_normalization_12/batchnorm/mul/ReadVariableOp
1functional_3/batch_normalization_12/batchnorm/mulMul7functional_3/batch_normalization_12/batchnorm/Rsqrt:y:0Hfunctional_3/batch_normalization_12/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:23
1functional_3/batch_normalization_12/batchnorm/mul
3functional_3/batch_normalization_12/batchnorm/mul_1Mul(functional_3/conv1d_2/Relu:activations:05functional_3/batch_normalization_12/batchnorm/mul:z:0*
T0*+
_output_shapes
:??????????25
3functional_3/batch_normalization_12/batchnorm/mul_1
>functional_3/batch_normalization_12/batchnorm/ReadVariableOp_1ReadVariableOpGfunctional_3_batch_normalization_12_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02@
>functional_3/batch_normalization_12/batchnorm/ReadVariableOp_1
3functional_3/batch_normalization_12/batchnorm/mul_2MulFfunctional_3/batch_normalization_12/batchnorm/ReadVariableOp_1:value:05functional_3/batch_normalization_12/batchnorm/mul:z:0*
T0*
_output_shapes
:25
3functional_3/batch_normalization_12/batchnorm/mul_2
>functional_3/batch_normalization_12/batchnorm/ReadVariableOp_2ReadVariableOpGfunctional_3_batch_normalization_12_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02@
>functional_3/batch_normalization_12/batchnorm/ReadVariableOp_2
1functional_3/batch_normalization_12/batchnorm/subSubFfunctional_3/batch_normalization_12/batchnorm/ReadVariableOp_2:value:07functional_3/batch_normalization_12/batchnorm/mul_2:z:0*
T0*
_output_shapes
:23
1functional_3/batch_normalization_12/batchnorm/sub
3functional_3/batch_normalization_12/batchnorm/add_1AddV27functional_3/batch_normalization_12/batchnorm/mul_1:z:05functional_3/batch_normalization_12/batchnorm/sub:z:0*
T0*+
_output_shapes
:??????????25
3functional_3/batch_normalization_12/batchnorm/add_1Υ
-functional_3/dense_5/Tensordot/ReadVariableOpReadVariableOp6functional_3_dense_5_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype02/
-functional_3/dense_5/Tensordot/ReadVariableOp
#functional_3/dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#functional_3/dense_5/Tensordot/axes
#functional_3/dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#functional_3/dense_5/Tensordot/free³
$functional_3/dense_5/Tensordot/ShapeShape7functional_3/batch_normalization_12/batchnorm/add_1:z:0*
T0*
_output_shapes
:2&
$functional_3/dense_5/Tensordot/Shape
,functional_3/dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,functional_3/dense_5/Tensordot/GatherV2/axisΊ
'functional_3/dense_5/Tensordot/GatherV2GatherV2-functional_3/dense_5/Tensordot/Shape:output:0,functional_3/dense_5/Tensordot/free:output:05functional_3/dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'functional_3/dense_5/Tensordot/GatherV2’
.functional_3/dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.functional_3/dense_5/Tensordot/GatherV2_1/axisΐ
)functional_3/dense_5/Tensordot/GatherV2_1GatherV2-functional_3/dense_5/Tensordot/Shape:output:0,functional_3/dense_5/Tensordot/axes:output:07functional_3/dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)functional_3/dense_5/Tensordot/GatherV2_1
$functional_3/dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$functional_3/dense_5/Tensordot/ConstΤ
#functional_3/dense_5/Tensordot/ProdProd0functional_3/dense_5/Tensordot/GatherV2:output:0-functional_3/dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#functional_3/dense_5/Tensordot/Prod
&functional_3/dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&functional_3/dense_5/Tensordot/Const_1ά
%functional_3/dense_5/Tensordot/Prod_1Prod2functional_3/dense_5/Tensordot/GatherV2_1:output:0/functional_3/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%functional_3/dense_5/Tensordot/Prod_1
*functional_3/dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*functional_3/dense_5/Tensordot/concat/axis
%functional_3/dense_5/Tensordot/concatConcatV2,functional_3/dense_5/Tensordot/free:output:0,functional_3/dense_5/Tensordot/axes:output:03functional_3/dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%functional_3/dense_5/Tensordot/concatΰ
$functional_3/dense_5/Tensordot/stackPack,functional_3/dense_5/Tensordot/Prod:output:0.functional_3/dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$functional_3/dense_5/Tensordot/stack
(functional_3/dense_5/Tensordot/transpose	Transpose7functional_3/batch_normalization_12/batchnorm/add_1:z:0.functional_3/dense_5/Tensordot/concat:output:0*
T0*+
_output_shapes
:??????????2*
(functional_3/dense_5/Tensordot/transposeσ
&functional_3/dense_5/Tensordot/ReshapeReshape,functional_3/dense_5/Tensordot/transpose:y:0-functional_3/dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2(
&functional_3/dense_5/Tensordot/Reshapeς
%functional_3/dense_5/Tensordot/MatMulMatMul/functional_3/dense_5/Tensordot/Reshape:output:05functional_3/dense_5/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2'
%functional_3/dense_5/Tensordot/MatMul
&functional_3/dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2(
&functional_3/dense_5/Tensordot/Const_2
,functional_3/dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,functional_3/dense_5/Tensordot/concat_1/axis¦
'functional_3/dense_5/Tensordot/concat_1ConcatV20functional_3/dense_5/Tensordot/GatherV2:output:0/functional_3/dense_5/Tensordot/Const_2:output:05functional_3/dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'functional_3/dense_5/Tensordot/concat_1δ
functional_3/dense_5/TensordotReshape/functional_3/dense_5/Tensordot/MatMul:product:00functional_3/dense_5/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:??????????@2 
functional_3/dense_5/TensordotΛ
+functional_3/dense_5/BiasAdd/ReadVariableOpReadVariableOp4functional_3_dense_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+functional_3/dense_5/BiasAdd/ReadVariableOpΫ
functional_3/dense_5/BiasAddBiasAdd'functional_3/dense_5/Tensordot:output:03functional_3/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:??????????@2
functional_3/dense_5/BiasAdd
functional_3/dense_5/ReluRelu%functional_3/dense_5/BiasAdd:output:0*
T0*+
_output_shapes
:??????????@2
functional_3/dense_5/Reluώ
<functional_3/batch_normalization_13/batchnorm/ReadVariableOpReadVariableOpEfunctional_3_batch_normalization_13_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02>
<functional_3/batch_normalization_13/batchnorm/ReadVariableOp―
3functional_3/batch_normalization_13/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:25
3functional_3/batch_normalization_13/batchnorm/add/y
1functional_3/batch_normalization_13/batchnorm/addAddV2Dfunctional_3/batch_normalization_13/batchnorm/ReadVariableOp:value:0<functional_3/batch_normalization_13/batchnorm/add/y:output:0*
T0*
_output_shapes
:@23
1functional_3/batch_normalization_13/batchnorm/addΟ
3functional_3/batch_normalization_13/batchnorm/RsqrtRsqrt5functional_3/batch_normalization_13/batchnorm/add:z:0*
T0*
_output_shapes
:@25
3functional_3/batch_normalization_13/batchnorm/Rsqrt
@functional_3/batch_normalization_13/batchnorm/mul/ReadVariableOpReadVariableOpIfunctional_3_batch_normalization_13_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02B
@functional_3/batch_normalization_13/batchnorm/mul/ReadVariableOp
1functional_3/batch_normalization_13/batchnorm/mulMul7functional_3/batch_normalization_13/batchnorm/Rsqrt:y:0Hfunctional_3/batch_normalization_13/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@23
1functional_3/batch_normalization_13/batchnorm/mul
3functional_3/batch_normalization_13/batchnorm/mul_1Mul'functional_3/dense_5/Relu:activations:05functional_3/batch_normalization_13/batchnorm/mul:z:0*
T0*+
_output_shapes
:??????????@25
3functional_3/batch_normalization_13/batchnorm/mul_1
>functional_3/batch_normalization_13/batchnorm/ReadVariableOp_1ReadVariableOpGfunctional_3_batch_normalization_13_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02@
>functional_3/batch_normalization_13/batchnorm/ReadVariableOp_1
3functional_3/batch_normalization_13/batchnorm/mul_2MulFfunctional_3/batch_normalization_13/batchnorm/ReadVariableOp_1:value:05functional_3/batch_normalization_13/batchnorm/mul:z:0*
T0*
_output_shapes
:@25
3functional_3/batch_normalization_13/batchnorm/mul_2
>functional_3/batch_normalization_13/batchnorm/ReadVariableOp_2ReadVariableOpGfunctional_3_batch_normalization_13_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02@
>functional_3/batch_normalization_13/batchnorm/ReadVariableOp_2
1functional_3/batch_normalization_13/batchnorm/subSubFfunctional_3/batch_normalization_13/batchnorm/ReadVariableOp_2:value:07functional_3/batch_normalization_13/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@23
1functional_3/batch_normalization_13/batchnorm/sub
3functional_3/batch_normalization_13/batchnorm/add_1AddV27functional_3/batch_normalization_13/batchnorm/mul_1:z:05functional_3/batch_normalization_13/batchnorm/sub:z:0*
T0*+
_output_shapes
:??????????@25
3functional_3/batch_normalization_13/batchnorm/add_1₯
+functional_3/conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????2-
+functional_3/conv1d_3/conv1d/ExpandDims/dim
'functional_3/conv1d_3/conv1d/ExpandDims
ExpandDims7functional_3/batch_normalization_13/batchnorm/add_1:z:04functional_3/conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:??????????@2)
'functional_3/conv1d_3/conv1d/ExpandDimsϊ
8functional_3/conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpAfunctional_3_conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8functional_3/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp 
-functional_3/conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-functional_3/conv1d_3/conv1d/ExpandDims_1/dim
)functional_3/conv1d_3/conv1d/ExpandDims_1
ExpandDims@functional_3/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:06functional_3/conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2+
)functional_3/conv1d_3/conv1d/ExpandDims_1
functional_3/conv1d_3/conv1dConv2D0functional_3/conv1d_3/conv1d/ExpandDims:output:02functional_3/conv1d_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
functional_3/conv1d_3/conv1dΤ
$functional_3/conv1d_3/conv1d/SqueezeSqueeze%functional_3/conv1d_3/conv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

ύ????????2&
$functional_3/conv1d_3/conv1d/SqueezeΞ
,functional_3/conv1d_3/BiasAdd/ReadVariableOpReadVariableOp5functional_3_conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,functional_3/conv1d_3/BiasAdd/ReadVariableOpδ
functional_3/conv1d_3/BiasAddBiasAdd-functional_3/conv1d_3/conv1d/Squeeze:output:04functional_3/conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
functional_3/conv1d_3/BiasAdd
functional_3/conv1d_3/ReluRelu&functional_3/conv1d_3/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2
functional_3/conv1d_3/Reluώ
<functional_3/batch_normalization_14/batchnorm/ReadVariableOpReadVariableOpEfunctional_3_batch_normalization_14_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02>
<functional_3/batch_normalization_14/batchnorm/ReadVariableOp―
3functional_3/batch_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:25
3functional_3/batch_normalization_14/batchnorm/add/y
1functional_3/batch_normalization_14/batchnorm/addAddV2Dfunctional_3/batch_normalization_14/batchnorm/ReadVariableOp:value:0<functional_3/batch_normalization_14/batchnorm/add/y:output:0*
T0*
_output_shapes
:23
1functional_3/batch_normalization_14/batchnorm/addΟ
3functional_3/batch_normalization_14/batchnorm/RsqrtRsqrt5functional_3/batch_normalization_14/batchnorm/add:z:0*
T0*
_output_shapes
:25
3functional_3/batch_normalization_14/batchnorm/Rsqrt
@functional_3/batch_normalization_14/batchnorm/mul/ReadVariableOpReadVariableOpIfunctional_3_batch_normalization_14_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02B
@functional_3/batch_normalization_14/batchnorm/mul/ReadVariableOp
1functional_3/batch_normalization_14/batchnorm/mulMul7functional_3/batch_normalization_14/batchnorm/Rsqrt:y:0Hfunctional_3/batch_normalization_14/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:23
1functional_3/batch_normalization_14/batchnorm/mul
3functional_3/batch_normalization_14/batchnorm/mul_1Mul(functional_3/conv1d_3/Relu:activations:05functional_3/batch_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????25
3functional_3/batch_normalization_14/batchnorm/mul_1
>functional_3/batch_normalization_14/batchnorm/ReadVariableOp_1ReadVariableOpGfunctional_3_batch_normalization_14_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02@
>functional_3/batch_normalization_14/batchnorm/ReadVariableOp_1
3functional_3/batch_normalization_14/batchnorm/mul_2MulFfunctional_3/batch_normalization_14/batchnorm/ReadVariableOp_1:value:05functional_3/batch_normalization_14/batchnorm/mul:z:0*
T0*
_output_shapes
:25
3functional_3/batch_normalization_14/batchnorm/mul_2
>functional_3/batch_normalization_14/batchnorm/ReadVariableOp_2ReadVariableOpGfunctional_3_batch_normalization_14_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02@
>functional_3/batch_normalization_14/batchnorm/ReadVariableOp_2
1functional_3/batch_normalization_14/batchnorm/subSubFfunctional_3/batch_normalization_14/batchnorm/ReadVariableOp_2:value:07functional_3/batch_normalization_14/batchnorm/mul_2:z:0*
T0*
_output_shapes
:23
1functional_3/batch_normalization_14/batchnorm/sub
3functional_3/batch_normalization_14/batchnorm/add_1AddV27functional_3/batch_normalization_14/batchnorm/mul_1:z:05functional_3/batch_normalization_14/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????25
3functional_3/batch_normalization_14/batchnorm/add_1Υ
-functional_3/dense_6/Tensordot/ReadVariableOpReadVariableOp6functional_3_dense_6_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype02/
-functional_3/dense_6/Tensordot/ReadVariableOp
#functional_3/dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#functional_3/dense_6/Tensordot/axes
#functional_3/dense_6/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#functional_3/dense_6/Tensordot/free³
$functional_3/dense_6/Tensordot/ShapeShape7functional_3/batch_normalization_14/batchnorm/add_1:z:0*
T0*
_output_shapes
:2&
$functional_3/dense_6/Tensordot/Shape
,functional_3/dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,functional_3/dense_6/Tensordot/GatherV2/axisΊ
'functional_3/dense_6/Tensordot/GatherV2GatherV2-functional_3/dense_6/Tensordot/Shape:output:0,functional_3/dense_6/Tensordot/free:output:05functional_3/dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'functional_3/dense_6/Tensordot/GatherV2’
.functional_3/dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.functional_3/dense_6/Tensordot/GatherV2_1/axisΐ
)functional_3/dense_6/Tensordot/GatherV2_1GatherV2-functional_3/dense_6/Tensordot/Shape:output:0,functional_3/dense_6/Tensordot/axes:output:07functional_3/dense_6/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)functional_3/dense_6/Tensordot/GatherV2_1
$functional_3/dense_6/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$functional_3/dense_6/Tensordot/ConstΤ
#functional_3/dense_6/Tensordot/ProdProd0functional_3/dense_6/Tensordot/GatherV2:output:0-functional_3/dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#functional_3/dense_6/Tensordot/Prod
&functional_3/dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&functional_3/dense_6/Tensordot/Const_1ά
%functional_3/dense_6/Tensordot/Prod_1Prod2functional_3/dense_6/Tensordot/GatherV2_1:output:0/functional_3/dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%functional_3/dense_6/Tensordot/Prod_1
*functional_3/dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*functional_3/dense_6/Tensordot/concat/axis
%functional_3/dense_6/Tensordot/concatConcatV2,functional_3/dense_6/Tensordot/free:output:0,functional_3/dense_6/Tensordot/axes:output:03functional_3/dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%functional_3/dense_6/Tensordot/concatΰ
$functional_3/dense_6/Tensordot/stackPack,functional_3/dense_6/Tensordot/Prod:output:0.functional_3/dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$functional_3/dense_6/Tensordot/stack
(functional_3/dense_6/Tensordot/transpose	Transpose7functional_3/batch_normalization_14/batchnorm/add_1:z:0.functional_3/dense_6/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2*
(functional_3/dense_6/Tensordot/transposeσ
&functional_3/dense_6/Tensordot/ReshapeReshape,functional_3/dense_6/Tensordot/transpose:y:0-functional_3/dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2(
&functional_3/dense_6/Tensordot/Reshapeς
%functional_3/dense_6/Tensordot/MatMulMatMul/functional_3/dense_6/Tensordot/Reshape:output:05functional_3/dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2'
%functional_3/dense_6/Tensordot/MatMul
&functional_3/dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2(
&functional_3/dense_6/Tensordot/Const_2
,functional_3/dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,functional_3/dense_6/Tensordot/concat_1/axis¦
'functional_3/dense_6/Tensordot/concat_1ConcatV20functional_3/dense_6/Tensordot/GatherV2:output:0/functional_3/dense_6/Tensordot/Const_2:output:05functional_3/dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'functional_3/dense_6/Tensordot/concat_1δ
functional_3/dense_6/TensordotReshape/functional_3/dense_6/Tensordot/MatMul:product:00functional_3/dense_6/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????@2 
functional_3/dense_6/TensordotΛ
+functional_3/dense_6/BiasAdd/ReadVariableOpReadVariableOp4functional_3_dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+functional_3/dense_6/BiasAdd/ReadVariableOpΫ
functional_3/dense_6/BiasAddBiasAdd'functional_3/dense_6/Tensordot:output:03functional_3/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
functional_3/dense_6/BiasAdd
functional_3/dense_6/ReluRelu%functional_3/dense_6/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
functional_3/dense_6/Reluώ
<functional_3/batch_normalization_15/batchnorm/ReadVariableOpReadVariableOpEfunctional_3_batch_normalization_15_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02>
<functional_3/batch_normalization_15/batchnorm/ReadVariableOp―
3functional_3/batch_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:25
3functional_3/batch_normalization_15/batchnorm/add/y
1functional_3/batch_normalization_15/batchnorm/addAddV2Dfunctional_3/batch_normalization_15/batchnorm/ReadVariableOp:value:0<functional_3/batch_normalization_15/batchnorm/add/y:output:0*
T0*
_output_shapes
:@23
1functional_3/batch_normalization_15/batchnorm/addΟ
3functional_3/batch_normalization_15/batchnorm/RsqrtRsqrt5functional_3/batch_normalization_15/batchnorm/add:z:0*
T0*
_output_shapes
:@25
3functional_3/batch_normalization_15/batchnorm/Rsqrt
@functional_3/batch_normalization_15/batchnorm/mul/ReadVariableOpReadVariableOpIfunctional_3_batch_normalization_15_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02B
@functional_3/batch_normalization_15/batchnorm/mul/ReadVariableOp
1functional_3/batch_normalization_15/batchnorm/mulMul7functional_3/batch_normalization_15/batchnorm/Rsqrt:y:0Hfunctional_3/batch_normalization_15/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@23
1functional_3/batch_normalization_15/batchnorm/mul
3functional_3/batch_normalization_15/batchnorm/mul_1Mul'functional_3/dense_6/Relu:activations:05functional_3/batch_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????@25
3functional_3/batch_normalization_15/batchnorm/mul_1
>functional_3/batch_normalization_15/batchnorm/ReadVariableOp_1ReadVariableOpGfunctional_3_batch_normalization_15_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02@
>functional_3/batch_normalization_15/batchnorm/ReadVariableOp_1
3functional_3/batch_normalization_15/batchnorm/mul_2MulFfunctional_3/batch_normalization_15/batchnorm/ReadVariableOp_1:value:05functional_3/batch_normalization_15/batchnorm/mul:z:0*
T0*
_output_shapes
:@25
3functional_3/batch_normalization_15/batchnorm/mul_2
>functional_3/batch_normalization_15/batchnorm/ReadVariableOp_2ReadVariableOpGfunctional_3_batch_normalization_15_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02@
>functional_3/batch_normalization_15/batchnorm/ReadVariableOp_2
1functional_3/batch_normalization_15/batchnorm/subSubFfunctional_3/batch_normalization_15/batchnorm/ReadVariableOp_2:value:07functional_3/batch_normalization_15/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@23
1functional_3/batch_normalization_15/batchnorm/sub
3functional_3/batch_normalization_15/batchnorm/add_1AddV27functional_3/batch_normalization_15/batchnorm/mul_1:z:05functional_3/batch_normalization_15/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????@25
3functional_3/batch_normalization_15/batchnorm/add_1
functional_3/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
functional_3/flatten_1/Constή
functional_3/flatten_1/ReshapeReshape7functional_3/batch_normalization_15/batchnorm/add_1:z:0%functional_3/flatten_1/Const:output:0*
T0*(
_output_shapes
:?????????2 
functional_3/flatten_1/ReshapeΞ
*functional_3/dense_7/MatMul/ReadVariableOpReadVariableOp3functional_3_dense_7_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02,
*functional_3/dense_7/MatMul/ReadVariableOpΤ
functional_3/dense_7/MatMulMatMul'functional_3/flatten_1/Reshape:output:02functional_3/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
functional_3/dense_7/MatMulΜ
+functional_3/dense_7/BiasAdd/ReadVariableOpReadVariableOp4functional_3_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+functional_3/dense_7/BiasAdd/ReadVariableOpΦ
functional_3/dense_7/BiasAddBiasAdd%functional_3/dense_7/MatMul:product:03functional_3/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
functional_3/dense_7/BiasAdd
functional_3/dense_7/ReluRelu%functional_3/dense_7/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2
functional_3/dense_7/Relu?
<functional_3/batch_normalization_16/batchnorm/ReadVariableOpReadVariableOpEfunctional_3_batch_normalization_16_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02>
<functional_3/batch_normalization_16/batchnorm/ReadVariableOp―
3functional_3/batch_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:25
3functional_3/batch_normalization_16/batchnorm/add/y
1functional_3/batch_normalization_16/batchnorm/addAddV2Dfunctional_3/batch_normalization_16/batchnorm/ReadVariableOp:value:0<functional_3/batch_normalization_16/batchnorm/add/y:output:0*
T0*
_output_shapes	
:23
1functional_3/batch_normalization_16/batchnorm/addΠ
3functional_3/batch_normalization_16/batchnorm/RsqrtRsqrt5functional_3/batch_normalization_16/batchnorm/add:z:0*
T0*
_output_shapes	
:25
3functional_3/batch_normalization_16/batchnorm/Rsqrt
@functional_3/batch_normalization_16/batchnorm/mul/ReadVariableOpReadVariableOpIfunctional_3_batch_normalization_16_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02B
@functional_3/batch_normalization_16/batchnorm/mul/ReadVariableOp
1functional_3/batch_normalization_16/batchnorm/mulMul7functional_3/batch_normalization_16/batchnorm/Rsqrt:y:0Hfunctional_3/batch_normalization_16/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:23
1functional_3/batch_normalization_16/batchnorm/mul
3functional_3/batch_normalization_16/batchnorm/mul_1Mul'functional_3/dense_7/Relu:activations:05functional_3/batch_normalization_16/batchnorm/mul:z:0*
T0*(
_output_shapes
:?????????25
3functional_3/batch_normalization_16/batchnorm/mul_1
>functional_3/batch_normalization_16/batchnorm/ReadVariableOp_1ReadVariableOpGfunctional_3_batch_normalization_16_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02@
>functional_3/batch_normalization_16/batchnorm/ReadVariableOp_1
3functional_3/batch_normalization_16/batchnorm/mul_2MulFfunctional_3/batch_normalization_16/batchnorm/ReadVariableOp_1:value:05functional_3/batch_normalization_16/batchnorm/mul:z:0*
T0*
_output_shapes	
:25
3functional_3/batch_normalization_16/batchnorm/mul_2
>functional_3/batch_normalization_16/batchnorm/ReadVariableOp_2ReadVariableOpGfunctional_3_batch_normalization_16_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02@
>functional_3/batch_normalization_16/batchnorm/ReadVariableOp_2
1functional_3/batch_normalization_16/batchnorm/subSubFfunctional_3/batch_normalization_16/batchnorm/ReadVariableOp_2:value:07functional_3/batch_normalization_16/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:23
1functional_3/batch_normalization_16/batchnorm/sub
3functional_3/batch_normalization_16/batchnorm/add_1AddV27functional_3/batch_normalization_16/batchnorm/mul_1:z:05functional_3/batch_normalization_16/batchnorm/sub:z:0*
T0*(
_output_shapes
:?????????25
3functional_3/batch_normalization_16/batchnorm/add_1Ν
*functional_3/dense_8/MatMul/ReadVariableOpReadVariableOp3functional_3_dense_8_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02,
*functional_3/dense_8/MatMul/ReadVariableOpγ
functional_3/dense_8/MatMulMatMul7functional_3/batch_normalization_16/batchnorm/add_1:z:02functional_3/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
functional_3/dense_8/MatMulΛ
+functional_3/dense_8/BiasAdd/ReadVariableOpReadVariableOp4functional_3_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+functional_3/dense_8/BiasAdd/ReadVariableOpΥ
functional_3/dense_8/BiasAddBiasAdd%functional_3/dense_8/MatMul:product:03functional_3/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
functional_3/dense_8/BiasAdd
functional_3/dense_8/ReluRelu%functional_3/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
functional_3/dense_8/Reluώ
<functional_3/batch_normalization_17/batchnorm/ReadVariableOpReadVariableOpEfunctional_3_batch_normalization_17_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02>
<functional_3/batch_normalization_17/batchnorm/ReadVariableOp―
3functional_3/batch_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:25
3functional_3/batch_normalization_17/batchnorm/add/y
1functional_3/batch_normalization_17/batchnorm/addAddV2Dfunctional_3/batch_normalization_17/batchnorm/ReadVariableOp:value:0<functional_3/batch_normalization_17/batchnorm/add/y:output:0*
T0*
_output_shapes
:@23
1functional_3/batch_normalization_17/batchnorm/addΟ
3functional_3/batch_normalization_17/batchnorm/RsqrtRsqrt5functional_3/batch_normalization_17/batchnorm/add:z:0*
T0*
_output_shapes
:@25
3functional_3/batch_normalization_17/batchnorm/Rsqrt
@functional_3/batch_normalization_17/batchnorm/mul/ReadVariableOpReadVariableOpIfunctional_3_batch_normalization_17_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02B
@functional_3/batch_normalization_17/batchnorm/mul/ReadVariableOp
1functional_3/batch_normalization_17/batchnorm/mulMul7functional_3/batch_normalization_17/batchnorm/Rsqrt:y:0Hfunctional_3/batch_normalization_17/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@23
1functional_3/batch_normalization_17/batchnorm/mul
3functional_3/batch_normalization_17/batchnorm/mul_1Mul'functional_3/dense_8/Relu:activations:05functional_3/batch_normalization_17/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????@25
3functional_3/batch_normalization_17/batchnorm/mul_1
>functional_3/batch_normalization_17/batchnorm/ReadVariableOp_1ReadVariableOpGfunctional_3_batch_normalization_17_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02@
>functional_3/batch_normalization_17/batchnorm/ReadVariableOp_1
3functional_3/batch_normalization_17/batchnorm/mul_2MulFfunctional_3/batch_normalization_17/batchnorm/ReadVariableOp_1:value:05functional_3/batch_normalization_17/batchnorm/mul:z:0*
T0*
_output_shapes
:@25
3functional_3/batch_normalization_17/batchnorm/mul_2
>functional_3/batch_normalization_17/batchnorm/ReadVariableOp_2ReadVariableOpGfunctional_3_batch_normalization_17_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02@
>functional_3/batch_normalization_17/batchnorm/ReadVariableOp_2
1functional_3/batch_normalization_17/batchnorm/subSubFfunctional_3/batch_normalization_17/batchnorm/ReadVariableOp_2:value:07functional_3/batch_normalization_17/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@23
1functional_3/batch_normalization_17/batchnorm/sub
3functional_3/batch_normalization_17/batchnorm/add_1AddV27functional_3/batch_normalization_17/batchnorm/mul_1:z:05functional_3/batch_normalization_17/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????@25
3functional_3/batch_normalization_17/batchnorm/add_1Μ
*functional_3/dense_9/MatMul/ReadVariableOpReadVariableOp3functional_3_dense_9_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02,
*functional_3/dense_9/MatMul/ReadVariableOpγ
functional_3/dense_9/MatMulMatMul7functional_3/batch_normalization_17/batchnorm/add_1:z:02functional_3/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
functional_3/dense_9/MatMulΛ
+functional_3/dense_9/BiasAdd/ReadVariableOpReadVariableOp4functional_3_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+functional_3/dense_9/BiasAdd/ReadVariableOpΥ
functional_3/dense_9/BiasAddBiasAdd%functional_3/dense_9/MatMul:product:03functional_3/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
functional_3/dense_9/BiasAdd
functional_3/dense_9/TanhTanh%functional_3/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
functional_3/dense_9/Tanh±
&functional_3/tf_op_layer_Mul_1/Mul_1/yConst*
_output_shapes
:*
dtype0*-
value$B""  @@  @@  @@  @@  @@  @@2(
&functional_3/tf_op_layer_Mul_1/Mul_1/yδ
$functional_3/tf_op_layer_Mul_1/Mul_1Mulfunctional_3/dense_9/Tanh:y:0/functional_3/tf_op_layer_Mul_1/Mul_1/y:output:0*
T0*
_cloned(*'
_output_shapes
:?????????2&
$functional_3/tf_op_layer_Mul_1/Mul_1|
IdentityIdentity(functional_3/tf_op_layer_Mul_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*
_input_shapesϋ
ψ:?????????:::::::::::::::::::::::::::::::::::::::::::::::::::::::::U Q
,
_output_shapes
:?????????
!
_user_specified_name	input_2
¦
ͺ
7__inference_batch_normalization_10_layer_call_fn_401838

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall»
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_3979032
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
Υ

R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_399055

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Λ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????@:::::X T
0
_output_shapes
:?????????@
 
_user_specified_nameinputs
ς
ͺ
7__inference_batch_normalization_14_layer_call_fn_402572

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_3984422
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
β
ͺ
7__inference_batch_normalization_10_layer_call_fn_401902

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCallͺ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_3990552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????@
 
_user_specified_nameinputs
ί
?
C__inference_dense_5_layer_call_and_return_conditional_losses_402292

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisΡ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axisΧ
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:??????????2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:??????????@2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:??????????@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:??????????@2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:??????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:??????????:::S O
+
_output_shapes
:??????????
 
_user_specified_nameinputs


R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_401812

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ά
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@:::::i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
ΰ
ͺ
7__inference_batch_normalization_10_layer_call_fn_401889

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall¨
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_3990372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????@
 
_user_specified_nameinputs
 )
Λ
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_403027

inputs
assignmovingavg_403002
assignmovingavg_1_403008)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity’#AssignMovingAvg/AssignSubVariableOp’%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@2
moments/StopGradient€
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????@2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices²
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze_1
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/403002*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_403002*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpΓ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/403002*
_output_shapes
:@2
AssignMovingAvg/subΊ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/403002*
_output_shapes
:@2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_403002AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/403002*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp€
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/403008*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_403008*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpΝ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/403008*
_output_shapes
:@2
AssignMovingAvg_1/subΔ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/403008*
_output_shapes
:@2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_403008AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/403008*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????@2
batchnorm/add_1΅
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
ή
©
6__inference_batch_normalization_9_layer_call_fn_401677

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_3989372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????@
 
_user_specified_nameinputs
α

R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_399155

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Π
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:?????????:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3r
IdentityIdentityFusedBatchNormV3:y:0*
T0*1
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:?????????:::::Y U
1
_output_shapes
:?????????
 
_user_specified_nameinputs
*
Λ
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_398269

inputs
assignmovingavg_398244
assignmovingavg_1_398250)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity’#AssignMovingAvg/AssignSubVariableOp’%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????@2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesΆ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze_1
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/398244*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_398244*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpΓ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/398244*
_output_shapes
:@2
AssignMovingAvg/subΊ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/398244*
_output_shapes
:@2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_398244AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/398244*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp€
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/398250*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_398250*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpΝ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/398250*
_output_shapes
:@2
AssignMovingAvg_1/subΔ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/398250*
_output_shapes
:@2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_398250AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/398250*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????@2
batchnorm/add_1Β
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*4
_output_shapes"
 :??????????????????@2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs

«
C__inference_dense_9_layer_call_and_return_conditional_losses_399891

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Tanh\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
π
ͺ
7__inference_batch_normalization_13_layer_call_fn_402370

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_3982692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????@2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs
ς
ͺ
7__inference_batch_normalization_12_layer_call_fn_402179

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_3981622
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
	
¬
D__inference_conv2d_3_layer_call_and_return_conditional_losses_401617

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp₯
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????@2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs

~
)__inference_conv2d_4_layer_call_fn_401774

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_3990022
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????@
 
_user_specified_nameinputs
γ
}
(__inference_dense_7_layer_call_fn_402889

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallω
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_3997672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
	
¬
D__inference_conv2d_4_layer_call_and_return_conditional_losses_399002

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp₯
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????@2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????@:::X T
0
_output_shapes
:?????????@
 
_user_specified_nameinputs
Ζ

R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_402153

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1t
IdentityIdentitybatchnorm/add_1:z:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????:::::\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs

Ή
D__inference_conv1d_3_layer_call_and_return_conditional_losses_402481

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:??????????@2
conv1d/ExpandDimsΈ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1·
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

ύ????????2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:??????????@:::S O
+
_output_shapes
:??????????@
 
_user_specified_nameinputs

«
C__inference_dense_9_layer_call_and_return_conditional_losses_403084

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Tanh\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
Υ)
Λ
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_402215

inputs
assignmovingavg_402190
assignmovingavg_1_402196)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity’#AssignMovingAvg/AssignSubVariableOp’%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient¨
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:??????????2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesΆ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/402190*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_402190*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpΓ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/402190*
_output_shapes
:2
AssignMovingAvg/subΊ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/402190*
_output_shapes
:2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_402190AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/402190*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp€
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/402196*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_402196*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpΝ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/402196*
_output_shapes
:2
AssignMovingAvg_1/subΔ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/402196*
_output_shapes
:2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_402196AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/402196*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:??????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:??????????2
batchnorm/add_1Ή
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*+
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':??????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:S O
+
_output_shapes
:??????????
 
_user_specified_nameinputs
Ζ

R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_398442

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1t
IdentityIdentitybatchnorm/add_1:z:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????:::::\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
΅
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_399748

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????@:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs


R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_399706

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:?????????@2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????@2
batchnorm/add_1k
IdentityIdentitybatchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????@:::::S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
α

R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_402024

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Π
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:?????????:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3r
IdentityIdentityFusedBatchNormV3:y:0*
T0*1
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:?????????:::::Y U
1
_output_shapes
:?????????
 
_user_specified_nameinputs


R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_397903

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ά
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@:::::i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
*
Λ
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_402133

inputs
assignmovingavg_402108
assignmovingavg_1_402114)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity’#AssignMovingAvg/AssignSubVariableOp’%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesΆ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/402108*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_402108*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpΓ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/402108*
_output_shapes
:2
AssignMovingAvg/subΊ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/402108*
_output_shapes
:2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_402108AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/402108*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp€
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/402114*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_402114*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpΝ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/402114*
_output_shapes
:2
AssignMovingAvg_1/subΔ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/402114*
_output_shapes
:2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_402114AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/402114*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1Β
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
Ζ

R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_402357

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????@2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????@2
batchnorm/add_1t
IdentityIdentitybatchnorm/add_1:z:0*
T0*4
_output_shapes"
 :??????????????????@2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????@:::::\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs
Ζ

R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_398302

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????@2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????@2
batchnorm/add_1t
IdentityIdentitybatchnorm/add_1:z:0*
T0*4
_output_shapes"
 :??????????????????@2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????@:::::\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs
Έ)
Λ
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_402925

inputs
assignmovingavg_402900
assignmovingavg_1_402906)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity’#AssignMovingAvg/AssignSubVariableOp’%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	2
moments/StopGradient₯
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:?????????2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices³
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze_1
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/402900*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_402900*
_output_shapes	
:*
dtype02 
AssignMovingAvg/ReadVariableOpΔ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/402900*
_output_shapes	
:2
AssignMovingAvg/sub»
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/402900*
_output_shapes	
:2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_402900AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/402900*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp€
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/402906*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_402906*
_output_shapes	
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpΞ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/402906*
_output_shapes	
:2
AssignMovingAvg_1/subΕ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/402906*
_output_shapes	
:2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_402906AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/402906*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:?????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:?????????2
batchnorm/add_1Ά
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
Κ
―
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_401794

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
ΐ
ͺ
7__inference_batch_normalization_16_layer_call_fn_402958

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_3986892
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
Κ
―
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_397872

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
Ι
?
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_397768

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
€
©
6__inference_batch_normalization_9_layer_call_fn_401754

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCallΊ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_3977992
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
΅
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_402864

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????@:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
ς
ͺ
7__inference_batch_normalization_13_layer_call_fn_402383

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_3983022
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????@2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs
	
¬
D__inference_conv2d_4_layer_call_and_return_conditional_losses_401765

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp₯
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????@2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????@:::X T
0
_output_shapes
:?????????@
 
_user_specified_nameinputs
ί
?
C__inference_dense_5_layer_call_and_return_conditional_losses_399374

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisΡ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axisΧ
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:??????????2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:??????????@2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:??????????@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:??????????@2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:??????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:??????????:::S O
+
_output_shapes
:??????????
 
_user_specified_nameinputs


R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_402945

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:?????????2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:?????????2
batchnorm/add_1h
IdentityIdentitybatchnorm/add_1:z:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::::P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
ΐΟ
ι
H__inference_functional_3_layer_call_and_return_conditional_losses_401361

inputs+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource1
-batch_normalization_9_readvariableop_resource3
/batch_normalization_9_readvariableop_1_resourceB
>batch_normalization_9_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource2
.batch_normalization_10_readvariableop_resource4
0batch_normalization_10_readvariableop_1_resourceC
?batch_normalization_10_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource2
.batch_normalization_11_readvariableop_resource4
0batch_normalization_11_readvariableop_1_resourceC
?batch_normalization_11_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource8
4conv1d_2_conv1d_expanddims_1_readvariableop_resource,
(conv1d_2_biasadd_readvariableop_resource<
8batch_normalization_12_batchnorm_readvariableop_resource@
<batch_normalization_12_batchnorm_mul_readvariableop_resource>
:batch_normalization_12_batchnorm_readvariableop_1_resource>
:batch_normalization_12_batchnorm_readvariableop_2_resource-
)dense_5_tensordot_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource<
8batch_normalization_13_batchnorm_readvariableop_resource@
<batch_normalization_13_batchnorm_mul_readvariableop_resource>
:batch_normalization_13_batchnorm_readvariableop_1_resource>
:batch_normalization_13_batchnorm_readvariableop_2_resource8
4conv1d_3_conv1d_expanddims_1_readvariableop_resource,
(conv1d_3_biasadd_readvariableop_resource<
8batch_normalization_14_batchnorm_readvariableop_resource@
<batch_normalization_14_batchnorm_mul_readvariableop_resource>
:batch_normalization_14_batchnorm_readvariableop_1_resource>
:batch_normalization_14_batchnorm_readvariableop_2_resource-
)dense_6_tensordot_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource<
8batch_normalization_15_batchnorm_readvariableop_resource@
<batch_normalization_15_batchnorm_mul_readvariableop_resource>
:batch_normalization_15_batchnorm_readvariableop_1_resource>
:batch_normalization_15_batchnorm_readvariableop_2_resource*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource<
8batch_normalization_16_batchnorm_readvariableop_resource@
<batch_normalization_16_batchnorm_mul_readvariableop_resource>
:batch_normalization_16_batchnorm_readvariableop_1_resource>
:batch_normalization_16_batchnorm_readvariableop_2_resource*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource<
8batch_normalization_17_batchnorm_readvariableop_resource@
<batch_normalization_17_batchnorm_mul_readvariableop_resource>
:batch_normalization_17_batchnorm_readvariableop_1_resource>
:batch_normalization_17_batchnorm_readvariableop_2_resource*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource
identity‘
)tf_op_layer_ExpandDims_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2+
)tf_op_layer_ExpandDims_1/ExpandDims_1/dimβ
%tf_op_layer_ExpandDims_1/ExpandDims_1
ExpandDimsinputs2tf_op_layer_ExpandDims_1/ExpandDims_1/dim:output:0*
T0*
_cloned(*0
_output_shapes
:?????????2'
%tf_op_layer_ExpandDims_1/ExpandDims_1°
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
conv2d_3/Conv2D/ReadVariableOpθ
conv2d_3/Conv2DConv2D.tf_op_layer_ExpandDims_1/ExpandDims_1:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_3/Conv2D§
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp­
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@2
conv2d_3/BiasAdd|
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@2
conv2d_3/ReluΆ
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_9/ReadVariableOpΌ
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_9/ReadVariableOp_1ι
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpο
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1δ
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_3/Relu:activations:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????@:@:@:@:@:*
epsilon%o:*
is_training( 2(
&batch_normalization_9/FusedBatchNormV3°
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_4/Conv2D/ReadVariableOpδ
conv2d_4/Conv2DConv2D*batch_normalization_9/FusedBatchNormV3:y:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_4/Conv2D§
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp­
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@2
conv2d_4/BiasAdd|
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@2
conv2d_4/ReluΉ
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_10/ReadVariableOpΏ
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_10/ReadVariableOp_1μ
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpς
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1κ
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_4/Relu:activations:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????@:@:@:@:@:*
epsilon%o:*
is_training( 2)
'batch_normalization_10/FusedBatchNormV3±
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02 
conv2d_5/Conv2D/ReadVariableOpζ
conv2d_5/Conv2DConv2D+batch_normalization_10/FusedBatchNormV3:y:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
2
conv2d_5/Conv2D¨
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????2
conv2d_5/BiasAdd}
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*1
_output_shapes
:?????????2
conv2d_5/ReluΊ
%batch_normalization_11/ReadVariableOpReadVariableOp.batch_normalization_11_readvariableop_resource*
_output_shapes	
:*
dtype02'
%batch_normalization_11/ReadVariableOpΐ
'batch_normalization_11/ReadVariableOp_1ReadVariableOp0batch_normalization_11_readvariableop_1_resource*
_output_shapes	
:*
dtype02)
'batch_normalization_11/ReadVariableOp_1ν
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype028
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpσ
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02:
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ο
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3conv2d_5/Relu:activations:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:?????????:::::*
epsilon%o:*
is_training( 2)
'batch_normalization_11/FusedBatchNormV3£
%tf_op_layer_Reshape_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2'
%tf_op_layer_Reshape_1/Reshape_1/shapeρ
tf_op_layer_Reshape_1/Reshape_1Reshape+batch_normalization_11/FusedBatchNormV3:y:0.tf_op_layer_Reshape_1/Reshape_1/shape:output:0*
T0*
_cloned(*-
_output_shapes
:?????????2!
tf_op_layer_Reshape_1/Reshape_1
max_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
max_pooling1d_1/ExpandDims/dimΥ
max_pooling1d_1/ExpandDims
ExpandDims(tf_op_layer_Reshape_1/Reshape_1:output:0'max_pooling1d_1/ExpandDims/dim:output:0*
T0*1
_output_shapes
:?????????2
max_pooling1d_1/ExpandDims?
max_pooling1d_1/MaxPoolMaxPool#max_pooling1d_1/ExpandDims:output:0*0
_output_shapes
:?????????*
ksize	
*
paddingVALID*
strides	
2
max_pooling1d_1/MaxPool­
max_pooling1d_1/SqueezeSqueeze max_pooling1d_1/MaxPool:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims
2
max_pooling1d_1/Squeeze©
(tf_op_layer_Transpose_1/Transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(tf_op_layer_Transpose_1/Transpose_1/permς
#tf_op_layer_Transpose_1/Transpose_1	Transpose max_pooling1d_1/Squeeze:output:01tf_op_layer_Transpose_1/Transpose_1/perm:output:0*
T0*
_cloned(*,
_output_shapes
:?????????2%
#tf_op_layer_Transpose_1/Transpose_1
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????2 
conv1d_2/conv1d/ExpandDims/dimΣ
conv1d_2/conv1d/ExpandDims
ExpandDims'tf_op_layer_Transpose_1/Transpose_1:y:0'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????2
conv1d_2/conv1d/ExpandDimsΣ
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dimΫ
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_2/conv1d/ExpandDims_1Ϋ
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:??????????*
paddingVALID*
strides
2
conv1d_2/conv1d­
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*+
_output_shapes
:??????????*
squeeze_dims

ύ????????2
conv1d_2/conv1d/Squeeze§
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv1d_2/BiasAdd/ReadVariableOp°
conv1d_2/BiasAddBiasAdd conv1d_2/conv1d/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:??????????2
conv1d_2/BiasAddw
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:??????????2
conv1d_2/ReluΧ
/batch_normalization_12/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_12_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_12/batchnorm/ReadVariableOp
&batch_normalization_12/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_12/batchnorm/add/yδ
$batch_normalization_12/batchnorm/addAddV27batch_normalization_12/batchnorm/ReadVariableOp:value:0/batch_normalization_12/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_12/batchnorm/add¨
&batch_normalization_12/batchnorm/RsqrtRsqrt(batch_normalization_12/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_12/batchnorm/Rsqrtγ
3batch_normalization_12/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_12_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_12/batchnorm/mul/ReadVariableOpα
$batch_normalization_12/batchnorm/mulMul*batch_normalization_12/batchnorm/Rsqrt:y:0;batch_normalization_12/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_12/batchnorm/mulΤ
&batch_normalization_12/batchnorm/mul_1Mulconv1d_2/Relu:activations:0(batch_normalization_12/batchnorm/mul:z:0*
T0*+
_output_shapes
:??????????2(
&batch_normalization_12/batchnorm/mul_1έ
1batch_normalization_12/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_12_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_12/batchnorm/ReadVariableOp_1α
&batch_normalization_12/batchnorm/mul_2Mul9batch_normalization_12/batchnorm/ReadVariableOp_1:value:0(batch_normalization_12/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_12/batchnorm/mul_2έ
1batch_normalization_12/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_12_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_12/batchnorm/ReadVariableOp_2ί
$batch_normalization_12/batchnorm/subSub9batch_normalization_12/batchnorm/ReadVariableOp_2:value:0*batch_normalization_12/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_12/batchnorm/subε
&batch_normalization_12/batchnorm/add_1AddV2*batch_normalization_12/batchnorm/mul_1:z:0(batch_normalization_12/batchnorm/sub:z:0*
T0*+
_output_shapes
:??????????2(
&batch_normalization_12/batchnorm/add_1?
 dense_5/Tensordot/ReadVariableOpReadVariableOp)dense_5_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype02"
 dense_5/Tensordot/ReadVariableOpz
dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_5/Tensordot/axes
dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_5/Tensordot/free
dense_5/Tensordot/ShapeShape*batch_normalization_12/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
dense_5/Tensordot/Shape
dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_5/Tensordot/GatherV2/axisω
dense_5/Tensordot/GatherV2GatherV2 dense_5/Tensordot/Shape:output:0dense_5/Tensordot/free:output:0(dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_5/Tensordot/GatherV2
!dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_5/Tensordot/GatherV2_1/axis?
dense_5/Tensordot/GatherV2_1GatherV2 dense_5/Tensordot/Shape:output:0dense_5/Tensordot/axes:output:0*dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_5/Tensordot/GatherV2_1|
dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_5/Tensordot/Const 
dense_5/Tensordot/ProdProd#dense_5/Tensordot/GatherV2:output:0 dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_5/Tensordot/Prod
dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_5/Tensordot/Const_1¨
dense_5/Tensordot/Prod_1Prod%dense_5/Tensordot/GatherV2_1:output:0"dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_5/Tensordot/Prod_1
dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_5/Tensordot/concat/axisΨ
dense_5/Tensordot/concatConcatV2dense_5/Tensordot/free:output:0dense_5/Tensordot/axes:output:0&dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_5/Tensordot/concat¬
dense_5/Tensordot/stackPackdense_5/Tensordot/Prod:output:0!dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_5/Tensordot/stackΜ
dense_5/Tensordot/transpose	Transpose*batch_normalization_12/batchnorm/add_1:z:0!dense_5/Tensordot/concat:output:0*
T0*+
_output_shapes
:??????????2
dense_5/Tensordot/transposeΏ
dense_5/Tensordot/ReshapeReshapedense_5/Tensordot/transpose:y:0 dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_5/Tensordot/ReshapeΎ
dense_5/Tensordot/MatMulMatMul"dense_5/Tensordot/Reshape:output:0(dense_5/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_5/Tensordot/MatMul
dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_5/Tensordot/Const_2
dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_5/Tensordot/concat_1/axisε
dense_5/Tensordot/concat_1ConcatV2#dense_5/Tensordot/GatherV2:output:0"dense_5/Tensordot/Const_2:output:0(dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_5/Tensordot/concat_1°
dense_5/TensordotReshape"dense_5/Tensordot/MatMul:product:0#dense_5/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:??????????@2
dense_5/Tensordot€
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_5/BiasAdd/ReadVariableOp§
dense_5/BiasAddBiasAdddense_5/Tensordot:output:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:??????????@2
dense_5/BiasAddt
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*+
_output_shapes
:??????????@2
dense_5/ReluΧ
/batch_normalization_13/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_13_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype021
/batch_normalization_13/batchnorm/ReadVariableOp
&batch_normalization_13/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_13/batchnorm/add/yδ
$batch_normalization_13/batchnorm/addAddV27batch_normalization_13/batchnorm/ReadVariableOp:value:0/batch_normalization_13/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2&
$batch_normalization_13/batchnorm/add¨
&batch_normalization_13/batchnorm/RsqrtRsqrt(batch_normalization_13/batchnorm/add:z:0*
T0*
_output_shapes
:@2(
&batch_normalization_13/batchnorm/Rsqrtγ
3batch_normalization_13/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_13_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype025
3batch_normalization_13/batchnorm/mul/ReadVariableOpα
$batch_normalization_13/batchnorm/mulMul*batch_normalization_13/batchnorm/Rsqrt:y:0;batch_normalization_13/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2&
$batch_normalization_13/batchnorm/mulΣ
&batch_normalization_13/batchnorm/mul_1Muldense_5/Relu:activations:0(batch_normalization_13/batchnorm/mul:z:0*
T0*+
_output_shapes
:??????????@2(
&batch_normalization_13/batchnorm/mul_1έ
1batch_normalization_13/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_13_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype023
1batch_normalization_13/batchnorm/ReadVariableOp_1α
&batch_normalization_13/batchnorm/mul_2Mul9batch_normalization_13/batchnorm/ReadVariableOp_1:value:0(batch_normalization_13/batchnorm/mul:z:0*
T0*
_output_shapes
:@2(
&batch_normalization_13/batchnorm/mul_2έ
1batch_normalization_13/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_13_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype023
1batch_normalization_13/batchnorm/ReadVariableOp_2ί
$batch_normalization_13/batchnorm/subSub9batch_normalization_13/batchnorm/ReadVariableOp_2:value:0*batch_normalization_13/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2&
$batch_normalization_13/batchnorm/subε
&batch_normalization_13/batchnorm/add_1AddV2*batch_normalization_13/batchnorm/mul_1:z:0(batch_normalization_13/batchnorm/sub:z:0*
T0*+
_output_shapes
:??????????@2(
&batch_normalization_13/batchnorm/add_1
conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????2 
conv1d_3/conv1d/ExpandDims/dimΥ
conv1d_3/conv1d/ExpandDims
ExpandDims*batch_normalization_13/batchnorm/add_1:z:0'conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:??????????@2
conv1d_3/conv1d/ExpandDimsΣ
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02-
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_3/conv1d/ExpandDims_1/dimΫ
conv1d_3/conv1d/ExpandDims_1
ExpandDims3conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d_3/conv1d/ExpandDims_1Ϋ
conv1d_3/conv1dConv2D#conv1d_3/conv1d/ExpandDims:output:0%conv1d_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d_3/conv1d­
conv1d_3/conv1d/SqueezeSqueezeconv1d_3/conv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

ύ????????2
conv1d_3/conv1d/Squeeze§
conv1d_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv1d_3/BiasAdd/ReadVariableOp°
conv1d_3/BiasAddBiasAdd conv1d_3/conv1d/Squeeze:output:0'conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
conv1d_3/BiasAddw
conv1d_3/ReluReluconv1d_3/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2
conv1d_3/ReluΧ
/batch_normalization_14/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_14_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_14/batchnorm/ReadVariableOp
&batch_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_14/batchnorm/add/yδ
$batch_normalization_14/batchnorm/addAddV27batch_normalization_14/batchnorm/ReadVariableOp:value:0/batch_normalization_14/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_14/batchnorm/add¨
&batch_normalization_14/batchnorm/RsqrtRsqrt(batch_normalization_14/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_14/batchnorm/Rsqrtγ
3batch_normalization_14/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_14_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_14/batchnorm/mul/ReadVariableOpα
$batch_normalization_14/batchnorm/mulMul*batch_normalization_14/batchnorm/Rsqrt:y:0;batch_normalization_14/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_14/batchnorm/mulΤ
&batch_normalization_14/batchnorm/mul_1Mulconv1d_3/Relu:activations:0(batch_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_14/batchnorm/mul_1έ
1batch_normalization_14/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_14_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_14/batchnorm/ReadVariableOp_1α
&batch_normalization_14/batchnorm/mul_2Mul9batch_normalization_14/batchnorm/ReadVariableOp_1:value:0(batch_normalization_14/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_14/batchnorm/mul_2έ
1batch_normalization_14/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_14_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_14/batchnorm/ReadVariableOp_2ί
$batch_normalization_14/batchnorm/subSub9batch_normalization_14/batchnorm/ReadVariableOp_2:value:0*batch_normalization_14/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_14/batchnorm/subε
&batch_normalization_14/batchnorm/add_1AddV2*batch_normalization_14/batchnorm/mul_1:z:0(batch_normalization_14/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_14/batchnorm/add_1?
 dense_6/Tensordot/ReadVariableOpReadVariableOp)dense_6_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype02"
 dense_6/Tensordot/ReadVariableOpz
dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_6/Tensordot/axes
dense_6/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_6/Tensordot/free
dense_6/Tensordot/ShapeShape*batch_normalization_14/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
dense_6/Tensordot/Shape
dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_6/Tensordot/GatherV2/axisω
dense_6/Tensordot/GatherV2GatherV2 dense_6/Tensordot/Shape:output:0dense_6/Tensordot/free:output:0(dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_6/Tensordot/GatherV2
!dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_6/Tensordot/GatherV2_1/axis?
dense_6/Tensordot/GatherV2_1GatherV2 dense_6/Tensordot/Shape:output:0dense_6/Tensordot/axes:output:0*dense_6/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_6/Tensordot/GatherV2_1|
dense_6/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_6/Tensordot/Const 
dense_6/Tensordot/ProdProd#dense_6/Tensordot/GatherV2:output:0 dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_6/Tensordot/Prod
dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_6/Tensordot/Const_1¨
dense_6/Tensordot/Prod_1Prod%dense_6/Tensordot/GatherV2_1:output:0"dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_6/Tensordot/Prod_1
dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_6/Tensordot/concat/axisΨ
dense_6/Tensordot/concatConcatV2dense_6/Tensordot/free:output:0dense_6/Tensordot/axes:output:0&dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_6/Tensordot/concat¬
dense_6/Tensordot/stackPackdense_6/Tensordot/Prod:output:0!dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_6/Tensordot/stackΜ
dense_6/Tensordot/transpose	Transpose*batch_normalization_14/batchnorm/add_1:z:0!dense_6/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2
dense_6/Tensordot/transposeΏ
dense_6/Tensordot/ReshapeReshapedense_6/Tensordot/transpose:y:0 dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_6/Tensordot/ReshapeΎ
dense_6/Tensordot/MatMulMatMul"dense_6/Tensordot/Reshape:output:0(dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_6/Tensordot/MatMul
dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_6/Tensordot/Const_2
dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_6/Tensordot/concat_1/axisε
dense_6/Tensordot/concat_1ConcatV2#dense_6/Tensordot/GatherV2:output:0"dense_6/Tensordot/Const_2:output:0(dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_6/Tensordot/concat_1°
dense_6/TensordotReshape"dense_6/Tensordot/MatMul:product:0#dense_6/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????@2
dense_6/Tensordot€
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_6/BiasAdd/ReadVariableOp§
dense_6/BiasAddBiasAdddense_6/Tensordot:output:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
dense_6/BiasAddt
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
dense_6/ReluΧ
/batch_normalization_15/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_15_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype021
/batch_normalization_15/batchnorm/ReadVariableOp
&batch_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_15/batchnorm/add/yδ
$batch_normalization_15/batchnorm/addAddV27batch_normalization_15/batchnorm/ReadVariableOp:value:0/batch_normalization_15/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2&
$batch_normalization_15/batchnorm/add¨
&batch_normalization_15/batchnorm/RsqrtRsqrt(batch_normalization_15/batchnorm/add:z:0*
T0*
_output_shapes
:@2(
&batch_normalization_15/batchnorm/Rsqrtγ
3batch_normalization_15/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_15_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype025
3batch_normalization_15/batchnorm/mul/ReadVariableOpα
$batch_normalization_15/batchnorm/mulMul*batch_normalization_15/batchnorm/Rsqrt:y:0;batch_normalization_15/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2&
$batch_normalization_15/batchnorm/mulΣ
&batch_normalization_15/batchnorm/mul_1Muldense_6/Relu:activations:0(batch_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????@2(
&batch_normalization_15/batchnorm/mul_1έ
1batch_normalization_15/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_15_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype023
1batch_normalization_15/batchnorm/ReadVariableOp_1α
&batch_normalization_15/batchnorm/mul_2Mul9batch_normalization_15/batchnorm/ReadVariableOp_1:value:0(batch_normalization_15/batchnorm/mul:z:0*
T0*
_output_shapes
:@2(
&batch_normalization_15/batchnorm/mul_2έ
1batch_normalization_15/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_15_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype023
1batch_normalization_15/batchnorm/ReadVariableOp_2ί
$batch_normalization_15/batchnorm/subSub9batch_normalization_15/batchnorm/ReadVariableOp_2:value:0*batch_normalization_15/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2&
$batch_normalization_15/batchnorm/subε
&batch_normalization_15/batchnorm/add_1AddV2*batch_normalization_15/batchnorm/mul_1:z:0(batch_normalization_15/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????@2(
&batch_normalization_15/batchnorm/add_1s
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
flatten_1/Constͺ
flatten_1/ReshapeReshape*batch_normalization_15/batchnorm/add_1:z:0flatten_1/Const:output:0*
T0*(
_output_shapes
:?????????2
flatten_1/Reshape§
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_7/MatMul/ReadVariableOp 
dense_7/MatMulMatMulflatten_1/Reshape:output:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
dense_7/MatMul₯
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_7/BiasAdd/ReadVariableOp’
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
dense_7/BiasAddq
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2
dense_7/ReluΨ
/batch_normalization_16/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_16_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype021
/batch_normalization_16/batchnorm/ReadVariableOp
&batch_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_16/batchnorm/add/yε
$batch_normalization_16/batchnorm/addAddV27batch_normalization_16/batchnorm/ReadVariableOp:value:0/batch_normalization_16/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2&
$batch_normalization_16/batchnorm/add©
&batch_normalization_16/batchnorm/RsqrtRsqrt(batch_normalization_16/batchnorm/add:z:0*
T0*
_output_shapes	
:2(
&batch_normalization_16/batchnorm/Rsqrtδ
3batch_normalization_16/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_16_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype025
3batch_normalization_16/batchnorm/mul/ReadVariableOpβ
$batch_normalization_16/batchnorm/mulMul*batch_normalization_16/batchnorm/Rsqrt:y:0;batch_normalization_16/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2&
$batch_normalization_16/batchnorm/mulΠ
&batch_normalization_16/batchnorm/mul_1Muldense_7/Relu:activations:0(batch_normalization_16/batchnorm/mul:z:0*
T0*(
_output_shapes
:?????????2(
&batch_normalization_16/batchnorm/mul_1ή
1batch_normalization_16/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_16_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype023
1batch_normalization_16/batchnorm/ReadVariableOp_1β
&batch_normalization_16/batchnorm/mul_2Mul9batch_normalization_16/batchnorm/ReadVariableOp_1:value:0(batch_normalization_16/batchnorm/mul:z:0*
T0*
_output_shapes	
:2(
&batch_normalization_16/batchnorm/mul_2ή
1batch_normalization_16/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_16_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype023
1batch_normalization_16/batchnorm/ReadVariableOp_2ΰ
$batch_normalization_16/batchnorm/subSub9batch_normalization_16/batchnorm/ReadVariableOp_2:value:0*batch_normalization_16/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2&
$batch_normalization_16/batchnorm/subβ
&batch_normalization_16/batchnorm/add_1AddV2*batch_normalization_16/batchnorm/mul_1:z:0(batch_normalization_16/batchnorm/sub:z:0*
T0*(
_output_shapes
:?????????2(
&batch_normalization_16/batchnorm/add_1¦
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
dense_8/MatMul/ReadVariableOp―
dense_8/MatMulMatMul*batch_normalization_16/batchnorm/add_1:z:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_8/MatMul€
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_8/BiasAdd/ReadVariableOp‘
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_8/BiasAddp
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_8/ReluΧ
/batch_normalization_17/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_17_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype021
/batch_normalization_17/batchnorm/ReadVariableOp
&batch_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_17/batchnorm/add/yδ
$batch_normalization_17/batchnorm/addAddV27batch_normalization_17/batchnorm/ReadVariableOp:value:0/batch_normalization_17/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2&
$batch_normalization_17/batchnorm/add¨
&batch_normalization_17/batchnorm/RsqrtRsqrt(batch_normalization_17/batchnorm/add:z:0*
T0*
_output_shapes
:@2(
&batch_normalization_17/batchnorm/Rsqrtγ
3batch_normalization_17/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_17_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype025
3batch_normalization_17/batchnorm/mul/ReadVariableOpα
$batch_normalization_17/batchnorm/mulMul*batch_normalization_17/batchnorm/Rsqrt:y:0;batch_normalization_17/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2&
$batch_normalization_17/batchnorm/mulΟ
&batch_normalization_17/batchnorm/mul_1Muldense_8/Relu:activations:0(batch_normalization_17/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????@2(
&batch_normalization_17/batchnorm/mul_1έ
1batch_normalization_17/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_17_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype023
1batch_normalization_17/batchnorm/ReadVariableOp_1α
&batch_normalization_17/batchnorm/mul_2Mul9batch_normalization_17/batchnorm/ReadVariableOp_1:value:0(batch_normalization_17/batchnorm/mul:z:0*
T0*
_output_shapes
:@2(
&batch_normalization_17/batchnorm/mul_2έ
1batch_normalization_17/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_17_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype023
1batch_normalization_17/batchnorm/ReadVariableOp_2ί
$batch_normalization_17/batchnorm/subSub9batch_normalization_17/batchnorm/ReadVariableOp_2:value:0*batch_normalization_17/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2&
$batch_normalization_17/batchnorm/subα
&batch_normalization_17/batchnorm/add_1AddV2*batch_normalization_17/batchnorm/mul_1:z:0(batch_normalization_17/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????@2(
&batch_normalization_17/batchnorm/add_1₯
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_9/MatMul/ReadVariableOp―
dense_9/MatMulMatMul*batch_normalization_17/batchnorm/add_1:z:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_9/MatMul€
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_9/BiasAdd/ReadVariableOp‘
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_9/BiasAddp
dense_9/TanhTanhdense_9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_9/Tanh
tf_op_layer_Mul_1/Mul_1/yConst*
_output_shapes
:*
dtype0*-
value$B""  @@  @@  @@  @@  @@  @@2
tf_op_layer_Mul_1/Mul_1/y°
tf_op_layer_Mul_1/Mul_1Muldense_9/Tanh:y:0"tf_op_layer_Mul_1/Mul_1/y:output:0*
T0*
_cloned(*'
_output_shapes
:?????????2
tf_op_layer_Mul_1/Mul_1o
IdentityIdentitytf_op_layer_Mul_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*
_input_shapesϋ
ψ:?????????:::::::::::::::::::::::::::::::::::::::::::::::::::::::::T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
	
¬
D__inference_conv2d_5_layer_call_and_return_conditional_losses_401913

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¦
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:?????????2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????@:::X T
0
_output_shapes
:?????????@
 
_user_specified_nameinputs


R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_402439

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:??????????@2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:??????????@2
batchnorm/add_1k
IdentityIdentitybatchnorm/add_1:z:0*
T0*+
_output_shapes
:??????????@2

Identity"
identityIdentity:output:0*:
_input_shapes)
':??????????@:::::S O
+
_output_shapes
:??????????@
 
_user_specified_nameinputs


R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_398862

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????@2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????@2
batchnorm/add_1g
IdentityIdentitybatchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@:::::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
*
Λ
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_398409

inputs
assignmovingavg_398384
assignmovingavg_1_398390)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity’#AssignMovingAvg/AssignSubVariableOp’%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesΆ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/398384*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_398384*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpΓ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/398384*
_output_shapes
:2
AssignMovingAvg/subΊ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/398384*
_output_shapes
:2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_398384AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/398384*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp€
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/398390*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_398390*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpΝ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/398390*
_output_shapes
:2
AssignMovingAvg_1/subΔ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/398390*
_output_shapes
:2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_398390AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/398390*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1Β
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
«
«
C__inference_dense_8_layer_call_and_return_conditional_losses_399829

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:::P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs

p
T__inference_tf_op_layer_ExpandDims_1_layer_call_and_return_conditional_losses_398883

inputs
identityo
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims_1/dim
ExpandDims_1
ExpandDimsinputsExpandDims_1/dim:output:0*
T0*
_cloned(*0
_output_shapes
:?????????2
ExpandDims_1r
IdentityIdentityExpandDims_1:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:?????????:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs

m
Q__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_402056

inputs
identityw
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_1/shape
	Reshape_1ReshapeinputsReshape_1/shape:output:0*
T0*
_cloned(*-
_output_shapes
:?????????2
	Reshape_1l
IdentityIdentityReshape_1:output:0*
T0*-
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:?????????:Y U
1
_output_shapes
:?????????
 
_user_specified_nameinputs

ϊ
H__inference_functional_3_layer_call_and_return_conditional_losses_400465

inputs
conv2d_3_400328
conv2d_3_400330 
batch_normalization_9_400333 
batch_normalization_9_400335 
batch_normalization_9_400337 
batch_normalization_9_400339
conv2d_4_400342
conv2d_4_400344!
batch_normalization_10_400347!
batch_normalization_10_400349!
batch_normalization_10_400351!
batch_normalization_10_400353
conv2d_5_400356
conv2d_5_400358!
batch_normalization_11_400361!
batch_normalization_11_400363!
batch_normalization_11_400365!
batch_normalization_11_400367
conv1d_2_400373
conv1d_2_400375!
batch_normalization_12_400378!
batch_normalization_12_400380!
batch_normalization_12_400382!
batch_normalization_12_400384
dense_5_400387
dense_5_400389!
batch_normalization_13_400392!
batch_normalization_13_400394!
batch_normalization_13_400396!
batch_normalization_13_400398
conv1d_3_400401
conv1d_3_400403!
batch_normalization_14_400406!
batch_normalization_14_400408!
batch_normalization_14_400410!
batch_normalization_14_400412
dense_6_400415
dense_6_400417!
batch_normalization_15_400420!
batch_normalization_15_400422!
batch_normalization_15_400424!
batch_normalization_15_400426
dense_7_400430
dense_7_400432!
batch_normalization_16_400435!
batch_normalization_16_400437!
batch_normalization_16_400439!
batch_normalization_16_400441
dense_8_400444
dense_8_400446!
batch_normalization_17_400449!
batch_normalization_17_400451!
batch_normalization_17_400453!
batch_normalization_17_400455
dense_9_400458
dense_9_400460
identity’.batch_normalization_10/StatefulPartitionedCall’.batch_normalization_11/StatefulPartitionedCall’.batch_normalization_12/StatefulPartitionedCall’.batch_normalization_13/StatefulPartitionedCall’.batch_normalization_14/StatefulPartitionedCall’.batch_normalization_15/StatefulPartitionedCall’.batch_normalization_16/StatefulPartitionedCall’.batch_normalization_17/StatefulPartitionedCall’-batch_normalization_9/StatefulPartitionedCall’ conv1d_2/StatefulPartitionedCall’ conv1d_3/StatefulPartitionedCall’ conv2d_3/StatefulPartitionedCall’ conv2d_4/StatefulPartitionedCall’ conv2d_5/StatefulPartitionedCall’dense_5/StatefulPartitionedCall’dense_6/StatefulPartitionedCall’dense_7/StatefulPartitionedCall’dense_8/StatefulPartitionedCall’dense_9/StatefulPartitionedCall
(tf_op_layer_ExpandDims_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_tf_op_layer_ExpandDims_1_layer_call_and_return_conditional_losses_3988832*
(tf_op_layer_ExpandDims_1/PartitionedCallΝ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall1tf_op_layer_ExpandDims_1/PartitionedCall:output:0conv2d_3_400328conv2d_3_400330*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_3989022"
 conv2d_3/StatefulPartitionedCallΖ
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_9_400333batch_normalization_9_400335batch_normalization_9_400337batch_normalization_9_400339*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_3989552/
-batch_normalization_9/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0conv2d_4_400342conv2d_4_400344*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_3990022"
 conv2d_4/StatefulPartitionedCallΝ
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_10_400347batch_normalization_10_400349batch_normalization_10_400351batch_normalization_10_400353*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_39905520
.batch_normalization_10/StatefulPartitionedCallΤ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0conv2d_5_400356conv2d_5_400358*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_3991022"
 conv2d_5/StatefulPartitionedCallΞ
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_11_400361batch_normalization_11_400363batch_normalization_11_400365batch_normalization_11_400367*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_39915520
.batch_normalization_11/StatefulPartitionedCall·
%tf_op_layer_Reshape_1/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_3991972'
%tf_op_layer_Reshape_1/PartitionedCall
max_pooling1d_1/PartitionedCallPartitionedCall.tf_op_layer_Reshape_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_3980272!
max_pooling1d_1/PartitionedCall­
'tf_op_layer_Transpose_1/PartitionedCallPartitionedCall(max_pooling1d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_tf_op_layer_Transpose_1_layer_call_and_return_conditional_losses_3992122)
'tf_op_layer_Transpose_1/PartitionedCallΗ
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall0tf_op_layer_Transpose_1/PartitionedCall:output:0conv1d_2_400373conv1d_2_400375*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_3992362"
 conv1d_2/StatefulPartitionedCallΘ
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0batch_normalization_12_400378batch_normalization_12_400380batch_normalization_12_400382batch_normalization_12_400384*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:??????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_39930720
.batch_normalization_12/StatefulPartitionedCallΙ
dense_5/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0dense_5_400387dense_5_400389*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:??????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_3993742!
dense_5/StatefulPartitionedCallΗ
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0batch_normalization_13_400392batch_normalization_13_400394batch_normalization_13_400396batch_normalization_13_400398*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:??????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_39944520
.batch_normalization_13/StatefulPartitionedCallΞ
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0conv1d_3_400401conv1d_3_400403*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv1d_3_layer_call_and_return_conditional_losses_3994972"
 conv1d_3/StatefulPartitionedCallΘ
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0batch_normalization_14_400406batch_normalization_14_400408batch_normalization_14_400410batch_normalization_14_400412*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_39956820
.batch_normalization_14/StatefulPartitionedCallΙ
dense_6/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0dense_6_400415dense_6_400417*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_3996352!
dense_6/StatefulPartitionedCallΗ
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0batch_normalization_15_400420batch_normalization_15_400422batch_normalization_15_400424batch_normalization_15_400426*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_39970620
.batch_normalization_15/StatefulPartitionedCall
flatten_1/PartitionedCallPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_3997482
flatten_1/PartitionedCall±
dense_7/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_7_400430dense_7_400432*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_3997672!
dense_7/StatefulPartitionedCallΔ
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0batch_normalization_16_400435batch_normalization_16_400437batch_normalization_16_400439batch_normalization_16_400441*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_39872220
.batch_normalization_16/StatefulPartitionedCallΕ
dense_8/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0dense_8_400444dense_8_400446*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_3998292!
dense_8/StatefulPartitionedCallΓ
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0batch_normalization_17_400449batch_normalization_17_400451batch_normalization_17_400453batch_normalization_17_400455*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_39886220
.batch_normalization_17/StatefulPartitionedCallΕ
dense_9/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0dense_9_400458dense_9_400460*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_3998912!
dense_9/StatefulPartitionedCall
!tf_op_layer_Mul_1/PartitionedCallPartitionedCall(dense_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_tf_op_layer_Mul_1_layer_call_and_return_conditional_losses_3999132#
!tf_op_layer_Mul_1/PartitionedCall
IdentityIdentity*tf_op_layer_Mul_1/PartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*
_input_shapesϋ
ψ:?????????::::::::::::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
Ξ
ͺ
7__inference_batch_normalization_15_layer_call_fn_402858

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall₯
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_3997062
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs

Ϊ
-__inference_functional_3_layer_call_fn_400322
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54
identity’StatefulPartitionedCallΧ
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54*D
Tin=
;29*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*H
_read_only_resource_inputs*
(&	
 #$%&)*+,/0125678*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_functional_3_layer_call_and_return_conditional_losses_4002072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*
_input_shapesϋ
ψ:?????????::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:?????????
!
_user_specified_name	input_2
Ξ
ͺ
7__inference_batch_normalization_13_layer_call_fn_402465

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall₯
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:??????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_3994452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:??????????@2

Identity"
identityIdentity:output:0*:
_input_shapes)
':??????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:??????????@
 
_user_specified_nameinputs
*
Λ
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_402526

inputs
assignmovingavg_402501
assignmovingavg_1_402507)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity’#AssignMovingAvg/AssignSubVariableOp’%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesΆ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/402501*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_402501*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpΓ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/402501*
_output_shapes
:2
AssignMovingAvg/subΊ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/402501*
_output_shapes
:2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_402501AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/402501*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp€
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/402507*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_402507*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpΝ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/402507*
_output_shapes
:2
AssignMovingAvg_1/subΔ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/402507*
_output_shapes
:2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_402507AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/402507*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1Β
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
π
ͺ
7__inference_batch_normalization_14_layer_call_fn_402559

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_3984092
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
Ό
ͺ
7__inference_batch_normalization_17_layer_call_fn_403060

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_3988292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
ΰ
©
6__inference_batch_normalization_9_layer_call_fn_401690

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_3989552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????@
 
_user_specified_nameinputs

ϋ
H__inference_functional_3_layer_call_and_return_conditional_losses_399922
input_2
conv2d_3_398913
conv2d_3_398915 
batch_normalization_9_398982 
batch_normalization_9_398984 
batch_normalization_9_398986 
batch_normalization_9_398988
conv2d_4_399013
conv2d_4_399015!
batch_normalization_10_399082!
batch_normalization_10_399084!
batch_normalization_10_399086!
batch_normalization_10_399088
conv2d_5_399113
conv2d_5_399115!
batch_normalization_11_399182!
batch_normalization_11_399184!
batch_normalization_11_399186!
batch_normalization_11_399188
conv1d_2_399247
conv1d_2_399249!
batch_normalization_12_399334!
batch_normalization_12_399336!
batch_normalization_12_399338!
batch_normalization_12_399340
dense_5_399385
dense_5_399387!
batch_normalization_13_399472!
batch_normalization_13_399474!
batch_normalization_13_399476!
batch_normalization_13_399478
conv1d_3_399508
conv1d_3_399510!
batch_normalization_14_399595!
batch_normalization_14_399597!
batch_normalization_14_399599!
batch_normalization_14_399601
dense_6_399646
dense_6_399648!
batch_normalization_15_399733!
batch_normalization_15_399735!
batch_normalization_15_399737!
batch_normalization_15_399739
dense_7_399778
dense_7_399780!
batch_normalization_16_399809!
batch_normalization_16_399811!
batch_normalization_16_399813!
batch_normalization_16_399815
dense_8_399840
dense_8_399842!
batch_normalization_17_399871!
batch_normalization_17_399873!
batch_normalization_17_399875!
batch_normalization_17_399877
dense_9_399902
dense_9_399904
identity’.batch_normalization_10/StatefulPartitionedCall’.batch_normalization_11/StatefulPartitionedCall’.batch_normalization_12/StatefulPartitionedCall’.batch_normalization_13/StatefulPartitionedCall’.batch_normalization_14/StatefulPartitionedCall’.batch_normalization_15/StatefulPartitionedCall’.batch_normalization_16/StatefulPartitionedCall’.batch_normalization_17/StatefulPartitionedCall’-batch_normalization_9/StatefulPartitionedCall’ conv1d_2/StatefulPartitionedCall’ conv1d_3/StatefulPartitionedCall’ conv2d_3/StatefulPartitionedCall’ conv2d_4/StatefulPartitionedCall’ conv2d_5/StatefulPartitionedCall’dense_5/StatefulPartitionedCall’dense_6/StatefulPartitionedCall’dense_7/StatefulPartitionedCall’dense_8/StatefulPartitionedCall’dense_9/StatefulPartitionedCall
(tf_op_layer_ExpandDims_1/PartitionedCallPartitionedCallinput_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_tf_op_layer_ExpandDims_1_layer_call_and_return_conditional_losses_3988832*
(tf_op_layer_ExpandDims_1/PartitionedCallΝ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall1tf_op_layer_ExpandDims_1/PartitionedCall:output:0conv2d_3_398913conv2d_3_398915*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_3989022"
 conv2d_3/StatefulPartitionedCallΔ
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_9_398982batch_normalization_9_398984batch_normalization_9_398986batch_normalization_9_398988*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_3989372/
-batch_normalization_9/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0conv2d_4_399013conv2d_4_399015*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_3990022"
 conv2d_4/StatefulPartitionedCallΛ
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_10_399082batch_normalization_10_399084batch_normalization_10_399086batch_normalization_10_399088*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_39903720
.batch_normalization_10/StatefulPartitionedCallΤ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0conv2d_5_399113conv2d_5_399115*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_3991022"
 conv2d_5/StatefulPartitionedCallΜ
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_11_399182batch_normalization_11_399184batch_normalization_11_399186batch_normalization_11_399188*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_39913720
.batch_normalization_11/StatefulPartitionedCall·
%tf_op_layer_Reshape_1/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_3991972'
%tf_op_layer_Reshape_1/PartitionedCall
max_pooling1d_1/PartitionedCallPartitionedCall.tf_op_layer_Reshape_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_3980272!
max_pooling1d_1/PartitionedCall­
'tf_op_layer_Transpose_1/PartitionedCallPartitionedCall(max_pooling1d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_tf_op_layer_Transpose_1_layer_call_and_return_conditional_losses_3992122)
'tf_op_layer_Transpose_1/PartitionedCallΗ
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall0tf_op_layer_Transpose_1/PartitionedCall:output:0conv1d_2_399247conv1d_2_399249*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_3992362"
 conv1d_2/StatefulPartitionedCallΖ
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0batch_normalization_12_399334batch_normalization_12_399336batch_normalization_12_399338batch_normalization_12_399340*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_39928720
.batch_normalization_12/StatefulPartitionedCallΙ
dense_5/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0dense_5_399385dense_5_399387*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:??????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_3993742!
dense_5/StatefulPartitionedCallΕ
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0batch_normalization_13_399472batch_normalization_13_399474batch_normalization_13_399476batch_normalization_13_399478*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:??????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_39942520
.batch_normalization_13/StatefulPartitionedCallΞ
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0conv1d_3_399508conv1d_3_399510*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv1d_3_layer_call_and_return_conditional_losses_3994972"
 conv1d_3/StatefulPartitionedCallΖ
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0batch_normalization_14_399595batch_normalization_14_399597batch_normalization_14_399599batch_normalization_14_399601*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_39954820
.batch_normalization_14/StatefulPartitionedCallΙ
dense_6/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0dense_6_399646dense_6_399648*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_3996352!
dense_6/StatefulPartitionedCallΕ
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0batch_normalization_15_399733batch_normalization_15_399735batch_normalization_15_399737batch_normalization_15_399739*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_39968620
.batch_normalization_15/StatefulPartitionedCall
flatten_1/PartitionedCallPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_3997482
flatten_1/PartitionedCall±
dense_7/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_7_399778dense_7_399780*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_3997672!
dense_7/StatefulPartitionedCallΒ
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0batch_normalization_16_399809batch_normalization_16_399811batch_normalization_16_399813batch_normalization_16_399815*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_39868920
.batch_normalization_16/StatefulPartitionedCallΕ
dense_8/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0dense_8_399840dense_8_399842*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_3998292!
dense_8/StatefulPartitionedCallΑ
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0batch_normalization_17_399871batch_normalization_17_399873batch_normalization_17_399875batch_normalization_17_399877*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_39882920
.batch_normalization_17/StatefulPartitionedCallΕ
dense_9/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0dense_9_399902dense_9_399904*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_3998912!
dense_9/StatefulPartitionedCall
!tf_op_layer_Mul_1/PartitionedCallPartitionedCall(dense_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_tf_op_layer_Mul_1_layer_call_and_return_conditional_losses_3999132#
!tf_op_layer_Mul_1/PartitionedCall
IdentityIdentity*tf_op_layer_Mul_1/PartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*
_input_shapesϋ
ψ:?????????::::::::::::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:U Q
,
_output_shapes
:?????????
!
_user_specified_name	input_2

Ή
D__inference_conv1d_3_layer_call_and_return_conditional_losses_399497

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:??????????@2
conv1d/ExpandDimsΈ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1·
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

ύ????????2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:??????????@:::S O
+
_output_shapes
:??????????@
 
_user_specified_nameinputs

―
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_402006

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ή
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:?????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*1
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:?????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:Y U
1
_output_shapes
:?????????
 
_user_specified_nameinputs
*
Λ
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_402337

inputs
assignmovingavg_402312
assignmovingavg_1_402318)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity’#AssignMovingAvg/AssignSubVariableOp’%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????@2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesΆ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze_1
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/402312*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_402312*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpΓ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/402312*
_output_shapes
:@2
AssignMovingAvg/subΊ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/402312*
_output_shapes
:@2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_402312AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/402312*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp€
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/402318*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_402318*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpΝ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/402318*
_output_shapes
:@2
AssignMovingAvg_1/subΔ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/402318*
_output_shapes
:@2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_402318AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/402318*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????@2
batchnorm/add_1Β
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*4
_output_shapes"
 :??????????????????@2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs
π
ͺ
7__inference_batch_normalization_15_layer_call_fn_402763

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_3985492
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????@2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs

~
)__inference_conv2d_5_layer_call_fn_401922

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_3991022
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????@
 
_user_specified_nameinputs
Υ)
Λ
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_399686

inputs
assignmovingavg_399661
assignmovingavg_1_399667)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity’#AssignMovingAvg/AssignSubVariableOp’%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradient¨
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:?????????@2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesΆ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze_1
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/399661*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_399661*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpΓ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/399661*
_output_shapes
:@2
AssignMovingAvg/subΊ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/399661*
_output_shapes
:@2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_399661AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/399661*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp€
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/399667*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_399667*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpΝ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/399667*
_output_shapes
:@2
AssignMovingAvg_1/subΔ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/399667*
_output_shapes
:@2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_399667AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/399667*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:?????????@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????@2
batchnorm/add_1Ή
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*+
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs

―
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_401858

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ω
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:?????????@
 
_user_specified_nameinputs
ς
ͺ
7__inference_batch_normalization_15_layer_call_fn_402776

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_3985822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????@2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs
Τ

Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_398955

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Λ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????@:::::X T
0
_output_shapes
:?????????@
 
_user_specified_nameinputs
±
«
C__inference_dense_7_layer_call_and_return_conditional_losses_399767

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:::P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs

Ϊ
-__inference_functional_3_layer_call_fn_400580
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54
identity’StatefulPartitionedCallι
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54*D
Tin=
;29*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*Z
_read_only_resource_inputs<
:8	
 !"#$%&'()*+,-./012345678*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_functional_3_layer_call_and_return_conditional_losses_4004652
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*
_input_shapesϋ
ψ:?????????::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:?????????
!
_user_specified_name	input_2
?Θ
΅ 
H__inference_functional_3_layer_call_and_return_conditional_losses_401085

inputs+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource1
-batch_normalization_9_readvariableop_resource3
/batch_normalization_9_readvariableop_1_resourceB
>batch_normalization_9_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource2
.batch_normalization_10_readvariableop_resource4
0batch_normalization_10_readvariableop_1_resourceC
?batch_normalization_10_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource2
.batch_normalization_11_readvariableop_resource4
0batch_normalization_11_readvariableop_1_resourceC
?batch_normalization_11_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource8
4conv1d_2_conv1d_expanddims_1_readvariableop_resource,
(conv1d_2_biasadd_readvariableop_resource1
-batch_normalization_12_assignmovingavg_4008093
/batch_normalization_12_assignmovingavg_1_400815@
<batch_normalization_12_batchnorm_mul_readvariableop_resource<
8batch_normalization_12_batchnorm_readvariableop_resource-
)dense_5_tensordot_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource1
-batch_normalization_13_assignmovingavg_4008683
/batch_normalization_13_assignmovingavg_1_400874@
<batch_normalization_13_batchnorm_mul_readvariableop_resource<
8batch_normalization_13_batchnorm_readvariableop_resource8
4conv1d_3_conv1d_expanddims_1_readvariableop_resource,
(conv1d_3_biasadd_readvariableop_resource1
-batch_normalization_14_assignmovingavg_4009123
/batch_normalization_14_assignmovingavg_1_400918@
<batch_normalization_14_batchnorm_mul_readvariableop_resource<
8batch_normalization_14_batchnorm_readvariableop_resource-
)dense_6_tensordot_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource1
-batch_normalization_15_assignmovingavg_4009713
/batch_normalization_15_assignmovingavg_1_400977@
<batch_normalization_15_batchnorm_mul_readvariableop_resource<
8batch_normalization_15_batchnorm_readvariableop_resource*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource1
-batch_normalization_16_assignmovingavg_4010123
/batch_normalization_16_assignmovingavg_1_401018@
<batch_normalization_16_batchnorm_mul_readvariableop_resource<
8batch_normalization_16_batchnorm_readvariableop_resource*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource1
-batch_normalization_17_assignmovingavg_4010513
/batch_normalization_17_assignmovingavg_1_401057@
<batch_normalization_17_batchnorm_mul_readvariableop_resource<
8batch_normalization_17_batchnorm_readvariableop_resource*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource
identity’%batch_normalization_10/AssignNewValue’'batch_normalization_10/AssignNewValue_1’%batch_normalization_11/AssignNewValue’'batch_normalization_11/AssignNewValue_1’:batch_normalization_12/AssignMovingAvg/AssignSubVariableOp’<batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOp’:batch_normalization_13/AssignMovingAvg/AssignSubVariableOp’<batch_normalization_13/AssignMovingAvg_1/AssignSubVariableOp’:batch_normalization_14/AssignMovingAvg/AssignSubVariableOp’<batch_normalization_14/AssignMovingAvg_1/AssignSubVariableOp’:batch_normalization_15/AssignMovingAvg/AssignSubVariableOp’<batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOp’:batch_normalization_16/AssignMovingAvg/AssignSubVariableOp’<batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOp’:batch_normalization_17/AssignMovingAvg/AssignSubVariableOp’<batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOp’$batch_normalization_9/AssignNewValue’&batch_normalization_9/AssignNewValue_1‘
)tf_op_layer_ExpandDims_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2+
)tf_op_layer_ExpandDims_1/ExpandDims_1/dimβ
%tf_op_layer_ExpandDims_1/ExpandDims_1
ExpandDimsinputs2tf_op_layer_ExpandDims_1/ExpandDims_1/dim:output:0*
T0*
_cloned(*0
_output_shapes
:?????????2'
%tf_op_layer_ExpandDims_1/ExpandDims_1°
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
conv2d_3/Conv2D/ReadVariableOpθ
conv2d_3/Conv2DConv2D.tf_op_layer_ExpandDims_1/ExpandDims_1:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_3/Conv2D§
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp­
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@2
conv2d_3/BiasAdd|
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@2
conv2d_3/ReluΆ
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_9/ReadVariableOpΌ
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_9/ReadVariableOp_1ι
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpο
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ς
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_3/Relu:activations:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
Χ#<2(
&batch_normalization_9/FusedBatchNormV3
$batch_normalization_9/AssignNewValueAssignVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource3batch_normalization_9/FusedBatchNormV3:batch_mean:06^batch_normalization_9/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_9/AssignNewValue
&batch_normalization_9/AssignNewValue_1AssignVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_9/FusedBatchNormV3:batch_variance:08^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_9/AssignNewValue_1°
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_4/Conv2D/ReadVariableOpδ
conv2d_4/Conv2DConv2D*batch_normalization_9/FusedBatchNormV3:y:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_4/Conv2D§
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp­
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@2
conv2d_4/BiasAdd|
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@2
conv2d_4/ReluΉ
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_10/ReadVariableOpΏ
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_10/ReadVariableOp_1μ
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpς
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ψ
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_4/Relu:activations:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
Χ#<2)
'batch_normalization_10/FusedBatchNormV3
%batch_normalization_10/AssignNewValueAssignVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource4batch_normalization_10/FusedBatchNormV3:batch_mean:07^batch_normalization_10/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_10/AssignNewValue
'batch_normalization_10/AssignNewValue_1AssignVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_10/FusedBatchNormV3:batch_variance:09^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_10/AssignNewValue_1±
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02 
conv2d_5/Conv2D/ReadVariableOpζ
conv2d_5/Conv2DConv2D+batch_normalization_10/FusedBatchNormV3:y:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
2
conv2d_5/Conv2D¨
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????2
conv2d_5/BiasAdd}
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*1
_output_shapes
:?????????2
conv2d_5/ReluΊ
%batch_normalization_11/ReadVariableOpReadVariableOp.batch_normalization_11_readvariableop_resource*
_output_shapes	
:*
dtype02'
%batch_normalization_11/ReadVariableOpΐ
'batch_normalization_11/ReadVariableOp_1ReadVariableOp0batch_normalization_11_readvariableop_1_resource*
_output_shapes	
:*
dtype02)
'batch_normalization_11/ReadVariableOp_1ν
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype028
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpσ
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02:
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ύ
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3conv2d_5/Relu:activations:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:?????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2)
'batch_normalization_11/FusedBatchNormV3
%batch_normalization_11/AssignNewValueAssignVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource4batch_normalization_11/FusedBatchNormV3:batch_mean:07^batch_normalization_11/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_11/AssignNewValue
'batch_normalization_11/AssignNewValue_1AssignVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_11/FusedBatchNormV3:batch_variance:09^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_11/AssignNewValue_1£
%tf_op_layer_Reshape_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2'
%tf_op_layer_Reshape_1/Reshape_1/shapeρ
tf_op_layer_Reshape_1/Reshape_1Reshape+batch_normalization_11/FusedBatchNormV3:y:0.tf_op_layer_Reshape_1/Reshape_1/shape:output:0*
T0*
_cloned(*-
_output_shapes
:?????????2!
tf_op_layer_Reshape_1/Reshape_1
max_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
max_pooling1d_1/ExpandDims/dimΥ
max_pooling1d_1/ExpandDims
ExpandDims(tf_op_layer_Reshape_1/Reshape_1:output:0'max_pooling1d_1/ExpandDims/dim:output:0*
T0*1
_output_shapes
:?????????2
max_pooling1d_1/ExpandDims?
max_pooling1d_1/MaxPoolMaxPool#max_pooling1d_1/ExpandDims:output:0*0
_output_shapes
:?????????*
ksize	
*
paddingVALID*
strides	
2
max_pooling1d_1/MaxPool­
max_pooling1d_1/SqueezeSqueeze max_pooling1d_1/MaxPool:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims
2
max_pooling1d_1/Squeeze©
(tf_op_layer_Transpose_1/Transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(tf_op_layer_Transpose_1/Transpose_1/permς
#tf_op_layer_Transpose_1/Transpose_1	Transpose max_pooling1d_1/Squeeze:output:01tf_op_layer_Transpose_1/Transpose_1/perm:output:0*
T0*
_cloned(*,
_output_shapes
:?????????2%
#tf_op_layer_Transpose_1/Transpose_1
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????2 
conv1d_2/conv1d/ExpandDims/dimΣ
conv1d_2/conv1d/ExpandDims
ExpandDims'tf_op_layer_Transpose_1/Transpose_1:y:0'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????2
conv1d_2/conv1d/ExpandDimsΣ
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dimΫ
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_2/conv1d/ExpandDims_1Ϋ
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:??????????*
paddingVALID*
strides
2
conv1d_2/conv1d­
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*+
_output_shapes
:??????????*
squeeze_dims

ύ????????2
conv1d_2/conv1d/Squeeze§
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv1d_2/BiasAdd/ReadVariableOp°
conv1d_2/BiasAddBiasAdd conv1d_2/conv1d/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:??????????2
conv1d_2/BiasAddw
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:??????????2
conv1d_2/ReluΏ
5batch_normalization_12/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       27
5batch_normalization_12/moments/mean/reduction_indicesν
#batch_normalization_12/moments/meanMeanconv1d_2/Relu:activations:0>batch_normalization_12/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2%
#batch_normalization_12/moments/meanΕ
+batch_normalization_12/moments/StopGradientStopGradient,batch_normalization_12/moments/mean:output:0*
T0*"
_output_shapes
:2-
+batch_normalization_12/moments/StopGradient
0batch_normalization_12/moments/SquaredDifferenceSquaredDifferenceconv1d_2/Relu:activations:04batch_normalization_12/moments/StopGradient:output:0*
T0*+
_output_shapes
:??????????22
0batch_normalization_12/moments/SquaredDifferenceΗ
9batch_normalization_12/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2;
9batch_normalization_12/moments/variance/reduction_indices
'batch_normalization_12/moments/varianceMean4batch_normalization_12/moments/SquaredDifference:z:0Bbatch_normalization_12/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2)
'batch_normalization_12/moments/varianceΖ
&batch_normalization_12/moments/SqueezeSqueeze,batch_normalization_12/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_12/moments/SqueezeΞ
(batch_normalization_12/moments/Squeeze_1Squeeze0batch_normalization_12/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_12/moments/Squeeze_1γ
,batch_normalization_12/AssignMovingAvg/decayConst*@
_class6
42loc:@batch_normalization_12/AssignMovingAvg/400809*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2.
,batch_normalization_12/AssignMovingAvg/decayΨ
5batch_normalization_12/AssignMovingAvg/ReadVariableOpReadVariableOp-batch_normalization_12_assignmovingavg_400809*
_output_shapes
:*
dtype027
5batch_normalization_12/AssignMovingAvg/ReadVariableOpΆ
*batch_normalization_12/AssignMovingAvg/subSub=batch_normalization_12/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_12/moments/Squeeze:output:0*
T0*@
_class6
42loc:@batch_normalization_12/AssignMovingAvg/400809*
_output_shapes
:2,
*batch_normalization_12/AssignMovingAvg/sub­
*batch_normalization_12/AssignMovingAvg/mulMul.batch_normalization_12/AssignMovingAvg/sub:z:05batch_normalization_12/AssignMovingAvg/decay:output:0*
T0*@
_class6
42loc:@batch_normalization_12/AssignMovingAvg/400809*
_output_shapes
:2,
*batch_normalization_12/AssignMovingAvg/mul
:batch_normalization_12/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-batch_normalization_12_assignmovingavg_400809.batch_normalization_12/AssignMovingAvg/mul:z:06^batch_normalization_12/AssignMovingAvg/ReadVariableOp*@
_class6
42loc:@batch_normalization_12/AssignMovingAvg/400809*
_output_shapes
 *
dtype02<
:batch_normalization_12/AssignMovingAvg/AssignSubVariableOpι
.batch_normalization_12/AssignMovingAvg_1/decayConst*B
_class8
64loc:@batch_normalization_12/AssignMovingAvg_1/400815*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<20
.batch_normalization_12/AssignMovingAvg_1/decayή
7batch_normalization_12/AssignMovingAvg_1/ReadVariableOpReadVariableOp/batch_normalization_12_assignmovingavg_1_400815*
_output_shapes
:*
dtype029
7batch_normalization_12/AssignMovingAvg_1/ReadVariableOpΐ
,batch_normalization_12/AssignMovingAvg_1/subSub?batch_normalization_12/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_12/moments/Squeeze_1:output:0*
T0*B
_class8
64loc:@batch_normalization_12/AssignMovingAvg_1/400815*
_output_shapes
:2.
,batch_normalization_12/AssignMovingAvg_1/sub·
,batch_normalization_12/AssignMovingAvg_1/mulMul0batch_normalization_12/AssignMovingAvg_1/sub:z:07batch_normalization_12/AssignMovingAvg_1/decay:output:0*
T0*B
_class8
64loc:@batch_normalization_12/AssignMovingAvg_1/400815*
_output_shapes
:2.
,batch_normalization_12/AssignMovingAvg_1/mul
<batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/batch_normalization_12_assignmovingavg_1_4008150batch_normalization_12/AssignMovingAvg_1/mul:z:08^batch_normalization_12/AssignMovingAvg_1/ReadVariableOp*B
_class8
64loc:@batch_normalization_12/AssignMovingAvg_1/400815*
_output_shapes
 *
dtype02>
<batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOp
&batch_normalization_12/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_12/batchnorm/add/yή
$batch_normalization_12/batchnorm/addAddV21batch_normalization_12/moments/Squeeze_1:output:0/batch_normalization_12/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_12/batchnorm/add¨
&batch_normalization_12/batchnorm/RsqrtRsqrt(batch_normalization_12/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_12/batchnorm/Rsqrtγ
3batch_normalization_12/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_12_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_12/batchnorm/mul/ReadVariableOpα
$batch_normalization_12/batchnorm/mulMul*batch_normalization_12/batchnorm/Rsqrt:y:0;batch_normalization_12/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_12/batchnorm/mulΤ
&batch_normalization_12/batchnorm/mul_1Mulconv1d_2/Relu:activations:0(batch_normalization_12/batchnorm/mul:z:0*
T0*+
_output_shapes
:??????????2(
&batch_normalization_12/batchnorm/mul_1Χ
&batch_normalization_12/batchnorm/mul_2Mul/batch_normalization_12/moments/Squeeze:output:0(batch_normalization_12/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_12/batchnorm/mul_2Χ
/batch_normalization_12/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_12_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_12/batchnorm/ReadVariableOpέ
$batch_normalization_12/batchnorm/subSub7batch_normalization_12/batchnorm/ReadVariableOp:value:0*batch_normalization_12/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_12/batchnorm/subε
&batch_normalization_12/batchnorm/add_1AddV2*batch_normalization_12/batchnorm/mul_1:z:0(batch_normalization_12/batchnorm/sub:z:0*
T0*+
_output_shapes
:??????????2(
&batch_normalization_12/batchnorm/add_1?
 dense_5/Tensordot/ReadVariableOpReadVariableOp)dense_5_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype02"
 dense_5/Tensordot/ReadVariableOpz
dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_5/Tensordot/axes
dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_5/Tensordot/free
dense_5/Tensordot/ShapeShape*batch_normalization_12/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
dense_5/Tensordot/Shape
dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_5/Tensordot/GatherV2/axisω
dense_5/Tensordot/GatherV2GatherV2 dense_5/Tensordot/Shape:output:0dense_5/Tensordot/free:output:0(dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_5/Tensordot/GatherV2
!dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_5/Tensordot/GatherV2_1/axis?
dense_5/Tensordot/GatherV2_1GatherV2 dense_5/Tensordot/Shape:output:0dense_5/Tensordot/axes:output:0*dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_5/Tensordot/GatherV2_1|
dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_5/Tensordot/Const 
dense_5/Tensordot/ProdProd#dense_5/Tensordot/GatherV2:output:0 dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_5/Tensordot/Prod
dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_5/Tensordot/Const_1¨
dense_5/Tensordot/Prod_1Prod%dense_5/Tensordot/GatherV2_1:output:0"dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_5/Tensordot/Prod_1
dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_5/Tensordot/concat/axisΨ
dense_5/Tensordot/concatConcatV2dense_5/Tensordot/free:output:0dense_5/Tensordot/axes:output:0&dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_5/Tensordot/concat¬
dense_5/Tensordot/stackPackdense_5/Tensordot/Prod:output:0!dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_5/Tensordot/stackΜ
dense_5/Tensordot/transpose	Transpose*batch_normalization_12/batchnorm/add_1:z:0!dense_5/Tensordot/concat:output:0*
T0*+
_output_shapes
:??????????2
dense_5/Tensordot/transposeΏ
dense_5/Tensordot/ReshapeReshapedense_5/Tensordot/transpose:y:0 dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_5/Tensordot/ReshapeΎ
dense_5/Tensordot/MatMulMatMul"dense_5/Tensordot/Reshape:output:0(dense_5/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_5/Tensordot/MatMul
dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_5/Tensordot/Const_2
dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_5/Tensordot/concat_1/axisε
dense_5/Tensordot/concat_1ConcatV2#dense_5/Tensordot/GatherV2:output:0"dense_5/Tensordot/Const_2:output:0(dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_5/Tensordot/concat_1°
dense_5/TensordotReshape"dense_5/Tensordot/MatMul:product:0#dense_5/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:??????????@2
dense_5/Tensordot€
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_5/BiasAdd/ReadVariableOp§
dense_5/BiasAddBiasAdddense_5/Tensordot:output:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:??????????@2
dense_5/BiasAddt
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*+
_output_shapes
:??????????@2
dense_5/ReluΏ
5batch_normalization_13/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       27
5batch_normalization_13/moments/mean/reduction_indicesμ
#batch_normalization_13/moments/meanMeandense_5/Relu:activations:0>batch_normalization_13/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2%
#batch_normalization_13/moments/meanΕ
+batch_normalization_13/moments/StopGradientStopGradient,batch_normalization_13/moments/mean:output:0*
T0*"
_output_shapes
:@2-
+batch_normalization_13/moments/StopGradient
0batch_normalization_13/moments/SquaredDifferenceSquaredDifferencedense_5/Relu:activations:04batch_normalization_13/moments/StopGradient:output:0*
T0*+
_output_shapes
:??????????@22
0batch_normalization_13/moments/SquaredDifferenceΗ
9batch_normalization_13/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2;
9batch_normalization_13/moments/variance/reduction_indices
'batch_normalization_13/moments/varianceMean4batch_normalization_13/moments/SquaredDifference:z:0Bbatch_normalization_13/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2)
'batch_normalization_13/moments/varianceΖ
&batch_normalization_13/moments/SqueezeSqueeze,batch_normalization_13/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2(
&batch_normalization_13/moments/SqueezeΞ
(batch_normalization_13/moments/Squeeze_1Squeeze0batch_normalization_13/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2*
(batch_normalization_13/moments/Squeeze_1γ
,batch_normalization_13/AssignMovingAvg/decayConst*@
_class6
42loc:@batch_normalization_13/AssignMovingAvg/400868*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2.
,batch_normalization_13/AssignMovingAvg/decayΨ
5batch_normalization_13/AssignMovingAvg/ReadVariableOpReadVariableOp-batch_normalization_13_assignmovingavg_400868*
_output_shapes
:@*
dtype027
5batch_normalization_13/AssignMovingAvg/ReadVariableOpΆ
*batch_normalization_13/AssignMovingAvg/subSub=batch_normalization_13/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_13/moments/Squeeze:output:0*
T0*@
_class6
42loc:@batch_normalization_13/AssignMovingAvg/400868*
_output_shapes
:@2,
*batch_normalization_13/AssignMovingAvg/sub­
*batch_normalization_13/AssignMovingAvg/mulMul.batch_normalization_13/AssignMovingAvg/sub:z:05batch_normalization_13/AssignMovingAvg/decay:output:0*
T0*@
_class6
42loc:@batch_normalization_13/AssignMovingAvg/400868*
_output_shapes
:@2,
*batch_normalization_13/AssignMovingAvg/mul
:batch_normalization_13/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-batch_normalization_13_assignmovingavg_400868.batch_normalization_13/AssignMovingAvg/mul:z:06^batch_normalization_13/AssignMovingAvg/ReadVariableOp*@
_class6
42loc:@batch_normalization_13/AssignMovingAvg/400868*
_output_shapes
 *
dtype02<
:batch_normalization_13/AssignMovingAvg/AssignSubVariableOpι
.batch_normalization_13/AssignMovingAvg_1/decayConst*B
_class8
64loc:@batch_normalization_13/AssignMovingAvg_1/400874*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<20
.batch_normalization_13/AssignMovingAvg_1/decayή
7batch_normalization_13/AssignMovingAvg_1/ReadVariableOpReadVariableOp/batch_normalization_13_assignmovingavg_1_400874*
_output_shapes
:@*
dtype029
7batch_normalization_13/AssignMovingAvg_1/ReadVariableOpΐ
,batch_normalization_13/AssignMovingAvg_1/subSub?batch_normalization_13/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_13/moments/Squeeze_1:output:0*
T0*B
_class8
64loc:@batch_normalization_13/AssignMovingAvg_1/400874*
_output_shapes
:@2.
,batch_normalization_13/AssignMovingAvg_1/sub·
,batch_normalization_13/AssignMovingAvg_1/mulMul0batch_normalization_13/AssignMovingAvg_1/sub:z:07batch_normalization_13/AssignMovingAvg_1/decay:output:0*
T0*B
_class8
64loc:@batch_normalization_13/AssignMovingAvg_1/400874*
_output_shapes
:@2.
,batch_normalization_13/AssignMovingAvg_1/mul
<batch_normalization_13/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/batch_normalization_13_assignmovingavg_1_4008740batch_normalization_13/AssignMovingAvg_1/mul:z:08^batch_normalization_13/AssignMovingAvg_1/ReadVariableOp*B
_class8
64loc:@batch_normalization_13/AssignMovingAvg_1/400874*
_output_shapes
 *
dtype02>
<batch_normalization_13/AssignMovingAvg_1/AssignSubVariableOp
&batch_normalization_13/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_13/batchnorm/add/yή
$batch_normalization_13/batchnorm/addAddV21batch_normalization_13/moments/Squeeze_1:output:0/batch_normalization_13/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2&
$batch_normalization_13/batchnorm/add¨
&batch_normalization_13/batchnorm/RsqrtRsqrt(batch_normalization_13/batchnorm/add:z:0*
T0*
_output_shapes
:@2(
&batch_normalization_13/batchnorm/Rsqrtγ
3batch_normalization_13/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_13_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype025
3batch_normalization_13/batchnorm/mul/ReadVariableOpα
$batch_normalization_13/batchnorm/mulMul*batch_normalization_13/batchnorm/Rsqrt:y:0;batch_normalization_13/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2&
$batch_normalization_13/batchnorm/mulΣ
&batch_normalization_13/batchnorm/mul_1Muldense_5/Relu:activations:0(batch_normalization_13/batchnorm/mul:z:0*
T0*+
_output_shapes
:??????????@2(
&batch_normalization_13/batchnorm/mul_1Χ
&batch_normalization_13/batchnorm/mul_2Mul/batch_normalization_13/moments/Squeeze:output:0(batch_normalization_13/batchnorm/mul:z:0*
T0*
_output_shapes
:@2(
&batch_normalization_13/batchnorm/mul_2Χ
/batch_normalization_13/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_13_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype021
/batch_normalization_13/batchnorm/ReadVariableOpέ
$batch_normalization_13/batchnorm/subSub7batch_normalization_13/batchnorm/ReadVariableOp:value:0*batch_normalization_13/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2&
$batch_normalization_13/batchnorm/subε
&batch_normalization_13/batchnorm/add_1AddV2*batch_normalization_13/batchnorm/mul_1:z:0(batch_normalization_13/batchnorm/sub:z:0*
T0*+
_output_shapes
:??????????@2(
&batch_normalization_13/batchnorm/add_1
conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????2 
conv1d_3/conv1d/ExpandDims/dimΥ
conv1d_3/conv1d/ExpandDims
ExpandDims*batch_normalization_13/batchnorm/add_1:z:0'conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:??????????@2
conv1d_3/conv1d/ExpandDimsΣ
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02-
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_3/conv1d/ExpandDims_1/dimΫ
conv1d_3/conv1d/ExpandDims_1
ExpandDims3conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d_3/conv1d/ExpandDims_1Ϋ
conv1d_3/conv1dConv2D#conv1d_3/conv1d/ExpandDims:output:0%conv1d_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d_3/conv1d­
conv1d_3/conv1d/SqueezeSqueezeconv1d_3/conv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

ύ????????2
conv1d_3/conv1d/Squeeze§
conv1d_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv1d_3/BiasAdd/ReadVariableOp°
conv1d_3/BiasAddBiasAdd conv1d_3/conv1d/Squeeze:output:0'conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
conv1d_3/BiasAddw
conv1d_3/ReluReluconv1d_3/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2
conv1d_3/ReluΏ
5batch_normalization_14/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       27
5batch_normalization_14/moments/mean/reduction_indicesν
#batch_normalization_14/moments/meanMeanconv1d_3/Relu:activations:0>batch_normalization_14/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2%
#batch_normalization_14/moments/meanΕ
+batch_normalization_14/moments/StopGradientStopGradient,batch_normalization_14/moments/mean:output:0*
T0*"
_output_shapes
:2-
+batch_normalization_14/moments/StopGradient
0batch_normalization_14/moments/SquaredDifferenceSquaredDifferenceconv1d_3/Relu:activations:04batch_normalization_14/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????22
0batch_normalization_14/moments/SquaredDifferenceΗ
9batch_normalization_14/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2;
9batch_normalization_14/moments/variance/reduction_indices
'batch_normalization_14/moments/varianceMean4batch_normalization_14/moments/SquaredDifference:z:0Bbatch_normalization_14/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2)
'batch_normalization_14/moments/varianceΖ
&batch_normalization_14/moments/SqueezeSqueeze,batch_normalization_14/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_14/moments/SqueezeΞ
(batch_normalization_14/moments/Squeeze_1Squeeze0batch_normalization_14/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_14/moments/Squeeze_1γ
,batch_normalization_14/AssignMovingAvg/decayConst*@
_class6
42loc:@batch_normalization_14/AssignMovingAvg/400912*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2.
,batch_normalization_14/AssignMovingAvg/decayΨ
5batch_normalization_14/AssignMovingAvg/ReadVariableOpReadVariableOp-batch_normalization_14_assignmovingavg_400912*
_output_shapes
:*
dtype027
5batch_normalization_14/AssignMovingAvg/ReadVariableOpΆ
*batch_normalization_14/AssignMovingAvg/subSub=batch_normalization_14/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_14/moments/Squeeze:output:0*
T0*@
_class6
42loc:@batch_normalization_14/AssignMovingAvg/400912*
_output_shapes
:2,
*batch_normalization_14/AssignMovingAvg/sub­
*batch_normalization_14/AssignMovingAvg/mulMul.batch_normalization_14/AssignMovingAvg/sub:z:05batch_normalization_14/AssignMovingAvg/decay:output:0*
T0*@
_class6
42loc:@batch_normalization_14/AssignMovingAvg/400912*
_output_shapes
:2,
*batch_normalization_14/AssignMovingAvg/mul
:batch_normalization_14/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-batch_normalization_14_assignmovingavg_400912.batch_normalization_14/AssignMovingAvg/mul:z:06^batch_normalization_14/AssignMovingAvg/ReadVariableOp*@
_class6
42loc:@batch_normalization_14/AssignMovingAvg/400912*
_output_shapes
 *
dtype02<
:batch_normalization_14/AssignMovingAvg/AssignSubVariableOpι
.batch_normalization_14/AssignMovingAvg_1/decayConst*B
_class8
64loc:@batch_normalization_14/AssignMovingAvg_1/400918*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<20
.batch_normalization_14/AssignMovingAvg_1/decayή
7batch_normalization_14/AssignMovingAvg_1/ReadVariableOpReadVariableOp/batch_normalization_14_assignmovingavg_1_400918*
_output_shapes
:*
dtype029
7batch_normalization_14/AssignMovingAvg_1/ReadVariableOpΐ
,batch_normalization_14/AssignMovingAvg_1/subSub?batch_normalization_14/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_14/moments/Squeeze_1:output:0*
T0*B
_class8
64loc:@batch_normalization_14/AssignMovingAvg_1/400918*
_output_shapes
:2.
,batch_normalization_14/AssignMovingAvg_1/sub·
,batch_normalization_14/AssignMovingAvg_1/mulMul0batch_normalization_14/AssignMovingAvg_1/sub:z:07batch_normalization_14/AssignMovingAvg_1/decay:output:0*
T0*B
_class8
64loc:@batch_normalization_14/AssignMovingAvg_1/400918*
_output_shapes
:2.
,batch_normalization_14/AssignMovingAvg_1/mul
<batch_normalization_14/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/batch_normalization_14_assignmovingavg_1_4009180batch_normalization_14/AssignMovingAvg_1/mul:z:08^batch_normalization_14/AssignMovingAvg_1/ReadVariableOp*B
_class8
64loc:@batch_normalization_14/AssignMovingAvg_1/400918*
_output_shapes
 *
dtype02>
<batch_normalization_14/AssignMovingAvg_1/AssignSubVariableOp
&batch_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_14/batchnorm/add/yή
$batch_normalization_14/batchnorm/addAddV21batch_normalization_14/moments/Squeeze_1:output:0/batch_normalization_14/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_14/batchnorm/add¨
&batch_normalization_14/batchnorm/RsqrtRsqrt(batch_normalization_14/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_14/batchnorm/Rsqrtγ
3batch_normalization_14/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_14_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_14/batchnorm/mul/ReadVariableOpα
$batch_normalization_14/batchnorm/mulMul*batch_normalization_14/batchnorm/Rsqrt:y:0;batch_normalization_14/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_14/batchnorm/mulΤ
&batch_normalization_14/batchnorm/mul_1Mulconv1d_3/Relu:activations:0(batch_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_14/batchnorm/mul_1Χ
&batch_normalization_14/batchnorm/mul_2Mul/batch_normalization_14/moments/Squeeze:output:0(batch_normalization_14/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_14/batchnorm/mul_2Χ
/batch_normalization_14/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_14_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_14/batchnorm/ReadVariableOpέ
$batch_normalization_14/batchnorm/subSub7batch_normalization_14/batchnorm/ReadVariableOp:value:0*batch_normalization_14/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_14/batchnorm/subε
&batch_normalization_14/batchnorm/add_1AddV2*batch_normalization_14/batchnorm/mul_1:z:0(batch_normalization_14/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_14/batchnorm/add_1?
 dense_6/Tensordot/ReadVariableOpReadVariableOp)dense_6_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype02"
 dense_6/Tensordot/ReadVariableOpz
dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_6/Tensordot/axes
dense_6/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_6/Tensordot/free
dense_6/Tensordot/ShapeShape*batch_normalization_14/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
dense_6/Tensordot/Shape
dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_6/Tensordot/GatherV2/axisω
dense_6/Tensordot/GatherV2GatherV2 dense_6/Tensordot/Shape:output:0dense_6/Tensordot/free:output:0(dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_6/Tensordot/GatherV2
!dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_6/Tensordot/GatherV2_1/axis?
dense_6/Tensordot/GatherV2_1GatherV2 dense_6/Tensordot/Shape:output:0dense_6/Tensordot/axes:output:0*dense_6/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_6/Tensordot/GatherV2_1|
dense_6/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_6/Tensordot/Const 
dense_6/Tensordot/ProdProd#dense_6/Tensordot/GatherV2:output:0 dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_6/Tensordot/Prod
dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_6/Tensordot/Const_1¨
dense_6/Tensordot/Prod_1Prod%dense_6/Tensordot/GatherV2_1:output:0"dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_6/Tensordot/Prod_1
dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_6/Tensordot/concat/axisΨ
dense_6/Tensordot/concatConcatV2dense_6/Tensordot/free:output:0dense_6/Tensordot/axes:output:0&dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_6/Tensordot/concat¬
dense_6/Tensordot/stackPackdense_6/Tensordot/Prod:output:0!dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_6/Tensordot/stackΜ
dense_6/Tensordot/transpose	Transpose*batch_normalization_14/batchnorm/add_1:z:0!dense_6/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2
dense_6/Tensordot/transposeΏ
dense_6/Tensordot/ReshapeReshapedense_6/Tensordot/transpose:y:0 dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_6/Tensordot/ReshapeΎ
dense_6/Tensordot/MatMulMatMul"dense_6/Tensordot/Reshape:output:0(dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_6/Tensordot/MatMul
dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_6/Tensordot/Const_2
dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_6/Tensordot/concat_1/axisε
dense_6/Tensordot/concat_1ConcatV2#dense_6/Tensordot/GatherV2:output:0"dense_6/Tensordot/Const_2:output:0(dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_6/Tensordot/concat_1°
dense_6/TensordotReshape"dense_6/Tensordot/MatMul:product:0#dense_6/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????@2
dense_6/Tensordot€
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_6/BiasAdd/ReadVariableOp§
dense_6/BiasAddBiasAdddense_6/Tensordot:output:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
dense_6/BiasAddt
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
dense_6/ReluΏ
5batch_normalization_15/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       27
5batch_normalization_15/moments/mean/reduction_indicesμ
#batch_normalization_15/moments/meanMeandense_6/Relu:activations:0>batch_normalization_15/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2%
#batch_normalization_15/moments/meanΕ
+batch_normalization_15/moments/StopGradientStopGradient,batch_normalization_15/moments/mean:output:0*
T0*"
_output_shapes
:@2-
+batch_normalization_15/moments/StopGradient
0batch_normalization_15/moments/SquaredDifferenceSquaredDifferencedense_6/Relu:activations:04batch_normalization_15/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????@22
0batch_normalization_15/moments/SquaredDifferenceΗ
9batch_normalization_15/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2;
9batch_normalization_15/moments/variance/reduction_indices
'batch_normalization_15/moments/varianceMean4batch_normalization_15/moments/SquaredDifference:z:0Bbatch_normalization_15/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2)
'batch_normalization_15/moments/varianceΖ
&batch_normalization_15/moments/SqueezeSqueeze,batch_normalization_15/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2(
&batch_normalization_15/moments/SqueezeΞ
(batch_normalization_15/moments/Squeeze_1Squeeze0batch_normalization_15/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2*
(batch_normalization_15/moments/Squeeze_1γ
,batch_normalization_15/AssignMovingAvg/decayConst*@
_class6
42loc:@batch_normalization_15/AssignMovingAvg/400971*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2.
,batch_normalization_15/AssignMovingAvg/decayΨ
5batch_normalization_15/AssignMovingAvg/ReadVariableOpReadVariableOp-batch_normalization_15_assignmovingavg_400971*
_output_shapes
:@*
dtype027
5batch_normalization_15/AssignMovingAvg/ReadVariableOpΆ
*batch_normalization_15/AssignMovingAvg/subSub=batch_normalization_15/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_15/moments/Squeeze:output:0*
T0*@
_class6
42loc:@batch_normalization_15/AssignMovingAvg/400971*
_output_shapes
:@2,
*batch_normalization_15/AssignMovingAvg/sub­
*batch_normalization_15/AssignMovingAvg/mulMul.batch_normalization_15/AssignMovingAvg/sub:z:05batch_normalization_15/AssignMovingAvg/decay:output:0*
T0*@
_class6
42loc:@batch_normalization_15/AssignMovingAvg/400971*
_output_shapes
:@2,
*batch_normalization_15/AssignMovingAvg/mul
:batch_normalization_15/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-batch_normalization_15_assignmovingavg_400971.batch_normalization_15/AssignMovingAvg/mul:z:06^batch_normalization_15/AssignMovingAvg/ReadVariableOp*@
_class6
42loc:@batch_normalization_15/AssignMovingAvg/400971*
_output_shapes
 *
dtype02<
:batch_normalization_15/AssignMovingAvg/AssignSubVariableOpι
.batch_normalization_15/AssignMovingAvg_1/decayConst*B
_class8
64loc:@batch_normalization_15/AssignMovingAvg_1/400977*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<20
.batch_normalization_15/AssignMovingAvg_1/decayή
7batch_normalization_15/AssignMovingAvg_1/ReadVariableOpReadVariableOp/batch_normalization_15_assignmovingavg_1_400977*
_output_shapes
:@*
dtype029
7batch_normalization_15/AssignMovingAvg_1/ReadVariableOpΐ
,batch_normalization_15/AssignMovingAvg_1/subSub?batch_normalization_15/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_15/moments/Squeeze_1:output:0*
T0*B
_class8
64loc:@batch_normalization_15/AssignMovingAvg_1/400977*
_output_shapes
:@2.
,batch_normalization_15/AssignMovingAvg_1/sub·
,batch_normalization_15/AssignMovingAvg_1/mulMul0batch_normalization_15/AssignMovingAvg_1/sub:z:07batch_normalization_15/AssignMovingAvg_1/decay:output:0*
T0*B
_class8
64loc:@batch_normalization_15/AssignMovingAvg_1/400977*
_output_shapes
:@2.
,batch_normalization_15/AssignMovingAvg_1/mul
<batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/batch_normalization_15_assignmovingavg_1_4009770batch_normalization_15/AssignMovingAvg_1/mul:z:08^batch_normalization_15/AssignMovingAvg_1/ReadVariableOp*B
_class8
64loc:@batch_normalization_15/AssignMovingAvg_1/400977*
_output_shapes
 *
dtype02>
<batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOp
&batch_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_15/batchnorm/add/yή
$batch_normalization_15/batchnorm/addAddV21batch_normalization_15/moments/Squeeze_1:output:0/batch_normalization_15/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2&
$batch_normalization_15/batchnorm/add¨
&batch_normalization_15/batchnorm/RsqrtRsqrt(batch_normalization_15/batchnorm/add:z:0*
T0*
_output_shapes
:@2(
&batch_normalization_15/batchnorm/Rsqrtγ
3batch_normalization_15/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_15_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype025
3batch_normalization_15/batchnorm/mul/ReadVariableOpα
$batch_normalization_15/batchnorm/mulMul*batch_normalization_15/batchnorm/Rsqrt:y:0;batch_normalization_15/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2&
$batch_normalization_15/batchnorm/mulΣ
&batch_normalization_15/batchnorm/mul_1Muldense_6/Relu:activations:0(batch_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????@2(
&batch_normalization_15/batchnorm/mul_1Χ
&batch_normalization_15/batchnorm/mul_2Mul/batch_normalization_15/moments/Squeeze:output:0(batch_normalization_15/batchnorm/mul:z:0*
T0*
_output_shapes
:@2(
&batch_normalization_15/batchnorm/mul_2Χ
/batch_normalization_15/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_15_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype021
/batch_normalization_15/batchnorm/ReadVariableOpέ
$batch_normalization_15/batchnorm/subSub7batch_normalization_15/batchnorm/ReadVariableOp:value:0*batch_normalization_15/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2&
$batch_normalization_15/batchnorm/subε
&batch_normalization_15/batchnorm/add_1AddV2*batch_normalization_15/batchnorm/mul_1:z:0(batch_normalization_15/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????@2(
&batch_normalization_15/batchnorm/add_1s
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
flatten_1/Constͺ
flatten_1/ReshapeReshape*batch_normalization_15/batchnorm/add_1:z:0flatten_1/Const:output:0*
T0*(
_output_shapes
:?????????2
flatten_1/Reshape§
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_7/MatMul/ReadVariableOp 
dense_7/MatMulMatMulflatten_1/Reshape:output:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
dense_7/MatMul₯
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_7/BiasAdd/ReadVariableOp’
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
dense_7/BiasAddq
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2
dense_7/ReluΈ
5batch_normalization_16/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_16/moments/mean/reduction_indicesι
#batch_normalization_16/moments/meanMeandense_7/Relu:activations:0>batch_normalization_16/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2%
#batch_normalization_16/moments/meanΒ
+batch_normalization_16/moments/StopGradientStopGradient,batch_normalization_16/moments/mean:output:0*
T0*
_output_shapes
:	2-
+batch_normalization_16/moments/StopGradientώ
0batch_normalization_16/moments/SquaredDifferenceSquaredDifferencedense_7/Relu:activations:04batch_normalization_16/moments/StopGradient:output:0*
T0*(
_output_shapes
:?????????22
0batch_normalization_16/moments/SquaredDifferenceΐ
9batch_normalization_16/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_16/moments/variance/reduction_indices
'batch_normalization_16/moments/varianceMean4batch_normalization_16/moments/SquaredDifference:z:0Bbatch_normalization_16/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2)
'batch_normalization_16/moments/varianceΖ
&batch_normalization_16/moments/SqueezeSqueeze,batch_normalization_16/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2(
&batch_normalization_16/moments/SqueezeΞ
(batch_normalization_16/moments/Squeeze_1Squeeze0batch_normalization_16/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2*
(batch_normalization_16/moments/Squeeze_1γ
,batch_normalization_16/AssignMovingAvg/decayConst*@
_class6
42loc:@batch_normalization_16/AssignMovingAvg/401012*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2.
,batch_normalization_16/AssignMovingAvg/decayΩ
5batch_normalization_16/AssignMovingAvg/ReadVariableOpReadVariableOp-batch_normalization_16_assignmovingavg_401012*
_output_shapes	
:*
dtype027
5batch_normalization_16/AssignMovingAvg/ReadVariableOp·
*batch_normalization_16/AssignMovingAvg/subSub=batch_normalization_16/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_16/moments/Squeeze:output:0*
T0*@
_class6
42loc:@batch_normalization_16/AssignMovingAvg/401012*
_output_shapes	
:2,
*batch_normalization_16/AssignMovingAvg/sub?
*batch_normalization_16/AssignMovingAvg/mulMul.batch_normalization_16/AssignMovingAvg/sub:z:05batch_normalization_16/AssignMovingAvg/decay:output:0*
T0*@
_class6
42loc:@batch_normalization_16/AssignMovingAvg/401012*
_output_shapes	
:2,
*batch_normalization_16/AssignMovingAvg/mul
:batch_normalization_16/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-batch_normalization_16_assignmovingavg_401012.batch_normalization_16/AssignMovingAvg/mul:z:06^batch_normalization_16/AssignMovingAvg/ReadVariableOp*@
_class6
42loc:@batch_normalization_16/AssignMovingAvg/401012*
_output_shapes
 *
dtype02<
:batch_normalization_16/AssignMovingAvg/AssignSubVariableOpι
.batch_normalization_16/AssignMovingAvg_1/decayConst*B
_class8
64loc:@batch_normalization_16/AssignMovingAvg_1/401018*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<20
.batch_normalization_16/AssignMovingAvg_1/decayί
7batch_normalization_16/AssignMovingAvg_1/ReadVariableOpReadVariableOp/batch_normalization_16_assignmovingavg_1_401018*
_output_shapes	
:*
dtype029
7batch_normalization_16/AssignMovingAvg_1/ReadVariableOpΑ
,batch_normalization_16/AssignMovingAvg_1/subSub?batch_normalization_16/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_16/moments/Squeeze_1:output:0*
T0*B
_class8
64loc:@batch_normalization_16/AssignMovingAvg_1/401018*
_output_shapes	
:2.
,batch_normalization_16/AssignMovingAvg_1/subΈ
,batch_normalization_16/AssignMovingAvg_1/mulMul0batch_normalization_16/AssignMovingAvg_1/sub:z:07batch_normalization_16/AssignMovingAvg_1/decay:output:0*
T0*B
_class8
64loc:@batch_normalization_16/AssignMovingAvg_1/401018*
_output_shapes	
:2.
,batch_normalization_16/AssignMovingAvg_1/mul
<batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/batch_normalization_16_assignmovingavg_1_4010180batch_normalization_16/AssignMovingAvg_1/mul:z:08^batch_normalization_16/AssignMovingAvg_1/ReadVariableOp*B
_class8
64loc:@batch_normalization_16/AssignMovingAvg_1/401018*
_output_shapes
 *
dtype02>
<batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOp
&batch_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_16/batchnorm/add/yί
$batch_normalization_16/batchnorm/addAddV21batch_normalization_16/moments/Squeeze_1:output:0/batch_normalization_16/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2&
$batch_normalization_16/batchnorm/add©
&batch_normalization_16/batchnorm/RsqrtRsqrt(batch_normalization_16/batchnorm/add:z:0*
T0*
_output_shapes	
:2(
&batch_normalization_16/batchnorm/Rsqrtδ
3batch_normalization_16/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_16_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype025
3batch_normalization_16/batchnorm/mul/ReadVariableOpβ
$batch_normalization_16/batchnorm/mulMul*batch_normalization_16/batchnorm/Rsqrt:y:0;batch_normalization_16/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2&
$batch_normalization_16/batchnorm/mulΠ
&batch_normalization_16/batchnorm/mul_1Muldense_7/Relu:activations:0(batch_normalization_16/batchnorm/mul:z:0*
T0*(
_output_shapes
:?????????2(
&batch_normalization_16/batchnorm/mul_1Ψ
&batch_normalization_16/batchnorm/mul_2Mul/batch_normalization_16/moments/Squeeze:output:0(batch_normalization_16/batchnorm/mul:z:0*
T0*
_output_shapes	
:2(
&batch_normalization_16/batchnorm/mul_2Ψ
/batch_normalization_16/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_16_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype021
/batch_normalization_16/batchnorm/ReadVariableOpή
$batch_normalization_16/batchnorm/subSub7batch_normalization_16/batchnorm/ReadVariableOp:value:0*batch_normalization_16/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2&
$batch_normalization_16/batchnorm/subβ
&batch_normalization_16/batchnorm/add_1AddV2*batch_normalization_16/batchnorm/mul_1:z:0(batch_normalization_16/batchnorm/sub:z:0*
T0*(
_output_shapes
:?????????2(
&batch_normalization_16/batchnorm/add_1¦
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
dense_8/MatMul/ReadVariableOp―
dense_8/MatMulMatMul*batch_normalization_16/batchnorm/add_1:z:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_8/MatMul€
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_8/BiasAdd/ReadVariableOp‘
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_8/BiasAddp
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_8/ReluΈ
5batch_normalization_17/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_17/moments/mean/reduction_indicesθ
#batch_normalization_17/moments/meanMeandense_8/Relu:activations:0>batch_normalization_17/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2%
#batch_normalization_17/moments/meanΑ
+batch_normalization_17/moments/StopGradientStopGradient,batch_normalization_17/moments/mean:output:0*
T0*
_output_shapes

:@2-
+batch_normalization_17/moments/StopGradientύ
0batch_normalization_17/moments/SquaredDifferenceSquaredDifferencedense_8/Relu:activations:04batch_normalization_17/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????@22
0batch_normalization_17/moments/SquaredDifferenceΐ
9batch_normalization_17/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_17/moments/variance/reduction_indices
'batch_normalization_17/moments/varianceMean4batch_normalization_17/moments/SquaredDifference:z:0Bbatch_normalization_17/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2)
'batch_normalization_17/moments/varianceΕ
&batch_normalization_17/moments/SqueezeSqueeze,batch_normalization_17/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2(
&batch_normalization_17/moments/SqueezeΝ
(batch_normalization_17/moments/Squeeze_1Squeeze0batch_normalization_17/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2*
(batch_normalization_17/moments/Squeeze_1γ
,batch_normalization_17/AssignMovingAvg/decayConst*@
_class6
42loc:@batch_normalization_17/AssignMovingAvg/401051*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2.
,batch_normalization_17/AssignMovingAvg/decayΨ
5batch_normalization_17/AssignMovingAvg/ReadVariableOpReadVariableOp-batch_normalization_17_assignmovingavg_401051*
_output_shapes
:@*
dtype027
5batch_normalization_17/AssignMovingAvg/ReadVariableOpΆ
*batch_normalization_17/AssignMovingAvg/subSub=batch_normalization_17/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_17/moments/Squeeze:output:0*
T0*@
_class6
42loc:@batch_normalization_17/AssignMovingAvg/401051*
_output_shapes
:@2,
*batch_normalization_17/AssignMovingAvg/sub­
*batch_normalization_17/AssignMovingAvg/mulMul.batch_normalization_17/AssignMovingAvg/sub:z:05batch_normalization_17/AssignMovingAvg/decay:output:0*
T0*@
_class6
42loc:@batch_normalization_17/AssignMovingAvg/401051*
_output_shapes
:@2,
*batch_normalization_17/AssignMovingAvg/mul
:batch_normalization_17/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-batch_normalization_17_assignmovingavg_401051.batch_normalization_17/AssignMovingAvg/mul:z:06^batch_normalization_17/AssignMovingAvg/ReadVariableOp*@
_class6
42loc:@batch_normalization_17/AssignMovingAvg/401051*
_output_shapes
 *
dtype02<
:batch_normalization_17/AssignMovingAvg/AssignSubVariableOpι
.batch_normalization_17/AssignMovingAvg_1/decayConst*B
_class8
64loc:@batch_normalization_17/AssignMovingAvg_1/401057*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<20
.batch_normalization_17/AssignMovingAvg_1/decayή
7batch_normalization_17/AssignMovingAvg_1/ReadVariableOpReadVariableOp/batch_normalization_17_assignmovingavg_1_401057*
_output_shapes
:@*
dtype029
7batch_normalization_17/AssignMovingAvg_1/ReadVariableOpΐ
,batch_normalization_17/AssignMovingAvg_1/subSub?batch_normalization_17/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_17/moments/Squeeze_1:output:0*
T0*B
_class8
64loc:@batch_normalization_17/AssignMovingAvg_1/401057*
_output_shapes
:@2.
,batch_normalization_17/AssignMovingAvg_1/sub·
,batch_normalization_17/AssignMovingAvg_1/mulMul0batch_normalization_17/AssignMovingAvg_1/sub:z:07batch_normalization_17/AssignMovingAvg_1/decay:output:0*
T0*B
_class8
64loc:@batch_normalization_17/AssignMovingAvg_1/401057*
_output_shapes
:@2.
,batch_normalization_17/AssignMovingAvg_1/mul
<batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/batch_normalization_17_assignmovingavg_1_4010570batch_normalization_17/AssignMovingAvg_1/mul:z:08^batch_normalization_17/AssignMovingAvg_1/ReadVariableOp*B
_class8
64loc:@batch_normalization_17/AssignMovingAvg_1/401057*
_output_shapes
 *
dtype02>
<batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOp
&batch_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_17/batchnorm/add/yή
$batch_normalization_17/batchnorm/addAddV21batch_normalization_17/moments/Squeeze_1:output:0/batch_normalization_17/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2&
$batch_normalization_17/batchnorm/add¨
&batch_normalization_17/batchnorm/RsqrtRsqrt(batch_normalization_17/batchnorm/add:z:0*
T0*
_output_shapes
:@2(
&batch_normalization_17/batchnorm/Rsqrtγ
3batch_normalization_17/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_17_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype025
3batch_normalization_17/batchnorm/mul/ReadVariableOpα
$batch_normalization_17/batchnorm/mulMul*batch_normalization_17/batchnorm/Rsqrt:y:0;batch_normalization_17/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2&
$batch_normalization_17/batchnorm/mulΟ
&batch_normalization_17/batchnorm/mul_1Muldense_8/Relu:activations:0(batch_normalization_17/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????@2(
&batch_normalization_17/batchnorm/mul_1Χ
&batch_normalization_17/batchnorm/mul_2Mul/batch_normalization_17/moments/Squeeze:output:0(batch_normalization_17/batchnorm/mul:z:0*
T0*
_output_shapes
:@2(
&batch_normalization_17/batchnorm/mul_2Χ
/batch_normalization_17/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_17_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype021
/batch_normalization_17/batchnorm/ReadVariableOpέ
$batch_normalization_17/batchnorm/subSub7batch_normalization_17/batchnorm/ReadVariableOp:value:0*batch_normalization_17/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2&
$batch_normalization_17/batchnorm/subα
&batch_normalization_17/batchnorm/add_1AddV2*batch_normalization_17/batchnorm/mul_1:z:0(batch_normalization_17/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????@2(
&batch_normalization_17/batchnorm/add_1₯
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_9/MatMul/ReadVariableOp―
dense_9/MatMulMatMul*batch_normalization_17/batchnorm/add_1:z:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_9/MatMul€
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_9/BiasAdd/ReadVariableOp‘
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_9/BiasAddp
dense_9/TanhTanhdense_9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_9/Tanh
tf_op_layer_Mul_1/Mul_1/yConst*
_output_shapes
:*
dtype0*-
value$B""  @@  @@  @@  @@  @@  @@2
tf_op_layer_Mul_1/Mul_1/y°
tf_op_layer_Mul_1/Mul_1Muldense_9/Tanh:y:0"tf_op_layer_Mul_1/Mul_1/y:output:0*
T0*
_cloned(*'
_output_shapes
:?????????2
tf_op_layer_Mul_1/Mul_1Λ
IdentityIdentitytf_op_layer_Mul_1/Mul_1:z:0&^batch_normalization_10/AssignNewValue(^batch_normalization_10/AssignNewValue_1&^batch_normalization_11/AssignNewValue(^batch_normalization_11/AssignNewValue_1;^batch_normalization_12/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_13/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_13/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_14/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_14/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_15/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_16/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_17/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOp%^batch_normalization_9/AssignNewValue'^batch_normalization_9/AssignNewValue_1*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*
_input_shapesϋ
ψ:?????????::::::::::::::::::::::::::::::::::::::::::::::::::::::::2N
%batch_normalization_10/AssignNewValue%batch_normalization_10/AssignNewValue2R
'batch_normalization_10/AssignNewValue_1'batch_normalization_10/AssignNewValue_12N
%batch_normalization_11/AssignNewValue%batch_normalization_11/AssignNewValue2R
'batch_normalization_11/AssignNewValue_1'batch_normalization_11/AssignNewValue_12x
:batch_normalization_12/AssignMovingAvg/AssignSubVariableOp:batch_normalization_12/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOp2x
:batch_normalization_13/AssignMovingAvg/AssignSubVariableOp:batch_normalization_13/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_13/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_13/AssignMovingAvg_1/AssignSubVariableOp2x
:batch_normalization_14/AssignMovingAvg/AssignSubVariableOp:batch_normalization_14/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_14/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_14/AssignMovingAvg_1/AssignSubVariableOp2x
:batch_normalization_15/AssignMovingAvg/AssignSubVariableOp:batch_normalization_15/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOp2x
:batch_normalization_16/AssignMovingAvg/AssignSubVariableOp:batch_normalization_16/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOp2x
:batch_normalization_17/AssignMovingAvg/AssignSubVariableOp:batch_normalization_17/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOp2L
$batch_normalization_9/AssignNewValue$batch_normalization_9/AssignNewValue2P
&batch_normalization_9/AssignNewValue_1&batch_normalization_9/AssignNewValue_1:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
Υ

R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_401876

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Λ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????@:::::X T
0
_output_shapes
:?????????@
 
_user_specified_nameinputs
ρ
~
)__inference_conv1d_3_layer_call_fn_402490

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallύ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv1d_3_layer_call_and_return_conditional_losses_3994972
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:??????????@::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:??????????@
 
_user_specified_nameinputs
Ι
T
8__inference_tf_op_layer_Transpose_1_layer_call_fn_402072

inputs
identityΫ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_tf_op_layer_Transpose_1_layer_call_and_return_conditional_losses_3992122
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:?????????:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs

p
T__inference_tf_op_layer_ExpandDims_1_layer_call_and_return_conditional_losses_401601

inputs
identityo
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims_1/dim
ExpandDims_1
ExpandDimsinputsExpandDims_1/dim:output:0*
T0*
_cloned(*0
_output_shapes
:?????????2
ExpandDims_1r
IdentityIdentityExpandDims_1:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:?????????:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs

o
S__inference_tf_op_layer_Transpose_1_layer_call_and_return_conditional_losses_402067

inputs
identityy
Transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
Transpose_1/perm
Transpose_1	TransposeinputsTranspose_1/perm:output:0*
T0*
_cloned(*,
_output_shapes
:?????????2
Transpose_1h
IdentityIdentityTranspose_1:y:0*
T0*,
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:?????????:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
Υ
ΰP
"__inference__traced_restore_403971
file_prefix$
 assignvariableop_conv2d_3_kernel$
 assignvariableop_1_conv2d_3_bias2
.assignvariableop_2_batch_normalization_9_gamma1
-assignvariableop_3_batch_normalization_9_beta8
4assignvariableop_4_batch_normalization_9_moving_mean<
8assignvariableop_5_batch_normalization_9_moving_variance&
"assignvariableop_6_conv2d_4_kernel$
 assignvariableop_7_conv2d_4_bias3
/assignvariableop_8_batch_normalization_10_gamma2
.assignvariableop_9_batch_normalization_10_beta:
6assignvariableop_10_batch_normalization_10_moving_mean>
:assignvariableop_11_batch_normalization_10_moving_variance'
#assignvariableop_12_conv2d_5_kernel%
!assignvariableop_13_conv2d_5_bias4
0assignvariableop_14_batch_normalization_11_gamma3
/assignvariableop_15_batch_normalization_11_beta:
6assignvariableop_16_batch_normalization_11_moving_mean>
:assignvariableop_17_batch_normalization_11_moving_variance'
#assignvariableop_18_conv1d_2_kernel%
!assignvariableop_19_conv1d_2_bias4
0assignvariableop_20_batch_normalization_12_gamma3
/assignvariableop_21_batch_normalization_12_beta:
6assignvariableop_22_batch_normalization_12_moving_mean>
:assignvariableop_23_batch_normalization_12_moving_variance&
"assignvariableop_24_dense_5_kernel$
 assignvariableop_25_dense_5_bias4
0assignvariableop_26_batch_normalization_13_gamma3
/assignvariableop_27_batch_normalization_13_beta:
6assignvariableop_28_batch_normalization_13_moving_mean>
:assignvariableop_29_batch_normalization_13_moving_variance'
#assignvariableop_30_conv1d_3_kernel%
!assignvariableop_31_conv1d_3_bias4
0assignvariableop_32_batch_normalization_14_gamma3
/assignvariableop_33_batch_normalization_14_beta:
6assignvariableop_34_batch_normalization_14_moving_mean>
:assignvariableop_35_batch_normalization_14_moving_variance&
"assignvariableop_36_dense_6_kernel$
 assignvariableop_37_dense_6_bias4
0assignvariableop_38_batch_normalization_15_gamma3
/assignvariableop_39_batch_normalization_15_beta:
6assignvariableop_40_batch_normalization_15_moving_mean>
:assignvariableop_41_batch_normalization_15_moving_variance&
"assignvariableop_42_dense_7_kernel$
 assignvariableop_43_dense_7_bias4
0assignvariableop_44_batch_normalization_16_gamma3
/assignvariableop_45_batch_normalization_16_beta:
6assignvariableop_46_batch_normalization_16_moving_mean>
:assignvariableop_47_batch_normalization_16_moving_variance&
"assignvariableop_48_dense_8_kernel$
 assignvariableop_49_dense_8_bias4
0assignvariableop_50_batch_normalization_17_gamma3
/assignvariableop_51_batch_normalization_17_beta:
6assignvariableop_52_batch_normalization_17_moving_mean>
:assignvariableop_53_batch_normalization_17_moving_variance&
"assignvariableop_54_dense_9_kernel$
 assignvariableop_55_dense_9_bias!
assignvariableop_56_adam_iter#
assignvariableop_57_adam_beta_1#
assignvariableop_58_adam_beta_2"
assignvariableop_59_adam_decay*
&assignvariableop_60_adam_learning_rate
assignvariableop_61_total
assignvariableop_62_count.
*assignvariableop_63_adam_conv2d_3_kernel_m,
(assignvariableop_64_adam_conv2d_3_bias_m:
6assignvariableop_65_adam_batch_normalization_9_gamma_m9
5assignvariableop_66_adam_batch_normalization_9_beta_m.
*assignvariableop_67_adam_conv2d_4_kernel_m,
(assignvariableop_68_adam_conv2d_4_bias_m;
7assignvariableop_69_adam_batch_normalization_10_gamma_m:
6assignvariableop_70_adam_batch_normalization_10_beta_m.
*assignvariableop_71_adam_conv2d_5_kernel_m,
(assignvariableop_72_adam_conv2d_5_bias_m;
7assignvariableop_73_adam_batch_normalization_11_gamma_m:
6assignvariableop_74_adam_batch_normalization_11_beta_m.
*assignvariableop_75_adam_conv1d_2_kernel_m,
(assignvariableop_76_adam_conv1d_2_bias_m;
7assignvariableop_77_adam_batch_normalization_12_gamma_m:
6assignvariableop_78_adam_batch_normalization_12_beta_m-
)assignvariableop_79_adam_dense_5_kernel_m+
'assignvariableop_80_adam_dense_5_bias_m;
7assignvariableop_81_adam_batch_normalization_13_gamma_m:
6assignvariableop_82_adam_batch_normalization_13_beta_m.
*assignvariableop_83_adam_conv1d_3_kernel_m,
(assignvariableop_84_adam_conv1d_3_bias_m;
7assignvariableop_85_adam_batch_normalization_14_gamma_m:
6assignvariableop_86_adam_batch_normalization_14_beta_m-
)assignvariableop_87_adam_dense_6_kernel_m+
'assignvariableop_88_adam_dense_6_bias_m;
7assignvariableop_89_adam_batch_normalization_15_gamma_m:
6assignvariableop_90_adam_batch_normalization_15_beta_m-
)assignvariableop_91_adam_dense_7_kernel_m+
'assignvariableop_92_adam_dense_7_bias_m;
7assignvariableop_93_adam_batch_normalization_16_gamma_m:
6assignvariableop_94_adam_batch_normalization_16_beta_m-
)assignvariableop_95_adam_dense_8_kernel_m+
'assignvariableop_96_adam_dense_8_bias_m;
7assignvariableop_97_adam_batch_normalization_17_gamma_m:
6assignvariableop_98_adam_batch_normalization_17_beta_m-
)assignvariableop_99_adam_dense_9_kernel_m,
(assignvariableop_100_adam_dense_9_bias_m/
+assignvariableop_101_adam_conv2d_3_kernel_v-
)assignvariableop_102_adam_conv2d_3_bias_v;
7assignvariableop_103_adam_batch_normalization_9_gamma_v:
6assignvariableop_104_adam_batch_normalization_9_beta_v/
+assignvariableop_105_adam_conv2d_4_kernel_v-
)assignvariableop_106_adam_conv2d_4_bias_v<
8assignvariableop_107_adam_batch_normalization_10_gamma_v;
7assignvariableop_108_adam_batch_normalization_10_beta_v/
+assignvariableop_109_adam_conv2d_5_kernel_v-
)assignvariableop_110_adam_conv2d_5_bias_v<
8assignvariableop_111_adam_batch_normalization_11_gamma_v;
7assignvariableop_112_adam_batch_normalization_11_beta_v/
+assignvariableop_113_adam_conv1d_2_kernel_v-
)assignvariableop_114_adam_conv1d_2_bias_v<
8assignvariableop_115_adam_batch_normalization_12_gamma_v;
7assignvariableop_116_adam_batch_normalization_12_beta_v.
*assignvariableop_117_adam_dense_5_kernel_v,
(assignvariableop_118_adam_dense_5_bias_v<
8assignvariableop_119_adam_batch_normalization_13_gamma_v;
7assignvariableop_120_adam_batch_normalization_13_beta_v/
+assignvariableop_121_adam_conv1d_3_kernel_v-
)assignvariableop_122_adam_conv1d_3_bias_v<
8assignvariableop_123_adam_batch_normalization_14_gamma_v;
7assignvariableop_124_adam_batch_normalization_14_beta_v.
*assignvariableop_125_adam_dense_6_kernel_v,
(assignvariableop_126_adam_dense_6_bias_v<
8assignvariableop_127_adam_batch_normalization_15_gamma_v;
7assignvariableop_128_adam_batch_normalization_15_beta_v.
*assignvariableop_129_adam_dense_7_kernel_v,
(assignvariableop_130_adam_dense_7_bias_v<
8assignvariableop_131_adam_batch_normalization_16_gamma_v;
7assignvariableop_132_adam_batch_normalization_16_beta_v.
*assignvariableop_133_adam_dense_8_kernel_v,
(assignvariableop_134_adam_dense_8_bias_v<
8assignvariableop_135_adam_batch_normalization_17_gamma_v;
7assignvariableop_136_adam_batch_normalization_17_beta_v.
*assignvariableop_137_adam_dense_9_kernel_v,
(assignvariableop_138_adam_dense_9_bias_v
identity_140’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_100’AssignVariableOp_101’AssignVariableOp_102’AssignVariableOp_103’AssignVariableOp_104’AssignVariableOp_105’AssignVariableOp_106’AssignVariableOp_107’AssignVariableOp_108’AssignVariableOp_109’AssignVariableOp_11’AssignVariableOp_110’AssignVariableOp_111’AssignVariableOp_112’AssignVariableOp_113’AssignVariableOp_114’AssignVariableOp_115’AssignVariableOp_116’AssignVariableOp_117’AssignVariableOp_118’AssignVariableOp_119’AssignVariableOp_12’AssignVariableOp_120’AssignVariableOp_121’AssignVariableOp_122’AssignVariableOp_123’AssignVariableOp_124’AssignVariableOp_125’AssignVariableOp_126’AssignVariableOp_127’AssignVariableOp_128’AssignVariableOp_129’AssignVariableOp_13’AssignVariableOp_130’AssignVariableOp_131’AssignVariableOp_132’AssignVariableOp_133’AssignVariableOp_134’AssignVariableOp_135’AssignVariableOp_136’AssignVariableOp_137’AssignVariableOp_138’AssignVariableOp_14’AssignVariableOp_15’AssignVariableOp_16’AssignVariableOp_17’AssignVariableOp_18’AssignVariableOp_19’AssignVariableOp_2’AssignVariableOp_20’AssignVariableOp_21’AssignVariableOp_22’AssignVariableOp_23’AssignVariableOp_24’AssignVariableOp_25’AssignVariableOp_26’AssignVariableOp_27’AssignVariableOp_28’AssignVariableOp_29’AssignVariableOp_3’AssignVariableOp_30’AssignVariableOp_31’AssignVariableOp_32’AssignVariableOp_33’AssignVariableOp_34’AssignVariableOp_35’AssignVariableOp_36’AssignVariableOp_37’AssignVariableOp_38’AssignVariableOp_39’AssignVariableOp_4’AssignVariableOp_40’AssignVariableOp_41’AssignVariableOp_42’AssignVariableOp_43’AssignVariableOp_44’AssignVariableOp_45’AssignVariableOp_46’AssignVariableOp_47’AssignVariableOp_48’AssignVariableOp_49’AssignVariableOp_5’AssignVariableOp_50’AssignVariableOp_51’AssignVariableOp_52’AssignVariableOp_53’AssignVariableOp_54’AssignVariableOp_55’AssignVariableOp_56’AssignVariableOp_57’AssignVariableOp_58’AssignVariableOp_59’AssignVariableOp_6’AssignVariableOp_60’AssignVariableOp_61’AssignVariableOp_62’AssignVariableOp_63’AssignVariableOp_64’AssignVariableOp_65’AssignVariableOp_66’AssignVariableOp_67’AssignVariableOp_68’AssignVariableOp_69’AssignVariableOp_7’AssignVariableOp_70’AssignVariableOp_71’AssignVariableOp_72’AssignVariableOp_73’AssignVariableOp_74’AssignVariableOp_75’AssignVariableOp_76’AssignVariableOp_77’AssignVariableOp_78’AssignVariableOp_79’AssignVariableOp_8’AssignVariableOp_80’AssignVariableOp_81’AssignVariableOp_82’AssignVariableOp_83’AssignVariableOp_84’AssignVariableOp_85’AssignVariableOp_86’AssignVariableOp_87’AssignVariableOp_88’AssignVariableOp_89’AssignVariableOp_9’AssignVariableOp_90’AssignVariableOp_91’AssignVariableOp_92’AssignVariableOp_93’AssignVariableOp_94’AssignVariableOp_95’AssignVariableOp_96’AssignVariableOp_97’AssignVariableOp_98’AssignVariableOp_99ΩN
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*δM
valueΪMBΧMB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-17/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-17/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names«
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*?
value€B‘B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesξ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ζ
_output_shapes³
°::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_conv2d_3_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1₯
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_3_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2³
AssignVariableOp_2AssignVariableOp.assignvariableop_2_batch_normalization_9_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3²
AssignVariableOp_3AssignVariableOp-assignvariableop_3_batch_normalization_9_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Ή
AssignVariableOp_4AssignVariableOp4assignvariableop_4_batch_normalization_9_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5½
AssignVariableOp_5AssignVariableOp8assignvariableop_5_batch_normalization_9_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6§
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_4_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7₯
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8΄
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_10_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9³
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_10_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ύ
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_10_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Β
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_10_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12«
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv2d_5_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13©
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv2d_5_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Έ
AssignVariableOp_14AssignVariableOp0assignvariableop_14_batch_normalization_11_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15·
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_11_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ύ
AssignVariableOp_16AssignVariableOp6assignvariableop_16_batch_normalization_11_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Β
AssignVariableOp_17AssignVariableOp:assignvariableop_17_batch_normalization_11_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18«
AssignVariableOp_18AssignVariableOp#assignvariableop_18_conv1d_2_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19©
AssignVariableOp_19AssignVariableOp!assignvariableop_19_conv1d_2_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Έ
AssignVariableOp_20AssignVariableOp0assignvariableop_20_batch_normalization_12_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21·
AssignVariableOp_21AssignVariableOp/assignvariableop_21_batch_normalization_12_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Ύ
AssignVariableOp_22AssignVariableOp6assignvariableop_22_batch_normalization_12_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Β
AssignVariableOp_23AssignVariableOp:assignvariableop_23_batch_normalization_12_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24ͺ
AssignVariableOp_24AssignVariableOp"assignvariableop_24_dense_5_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25¨
AssignVariableOp_25AssignVariableOp assignvariableop_25_dense_5_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Έ
AssignVariableOp_26AssignVariableOp0assignvariableop_26_batch_normalization_13_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27·
AssignVariableOp_27AssignVariableOp/assignvariableop_27_batch_normalization_13_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Ύ
AssignVariableOp_28AssignVariableOp6assignvariableop_28_batch_normalization_13_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Β
AssignVariableOp_29AssignVariableOp:assignvariableop_29_batch_normalization_13_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30«
AssignVariableOp_30AssignVariableOp#assignvariableop_30_conv1d_3_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31©
AssignVariableOp_31AssignVariableOp!assignvariableop_31_conv1d_3_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Έ
AssignVariableOp_32AssignVariableOp0assignvariableop_32_batch_normalization_14_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33·
AssignVariableOp_33AssignVariableOp/assignvariableop_33_batch_normalization_14_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Ύ
AssignVariableOp_34AssignVariableOp6assignvariableop_34_batch_normalization_14_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Β
AssignVariableOp_35AssignVariableOp:assignvariableop_35_batch_normalization_14_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36ͺ
AssignVariableOp_36AssignVariableOp"assignvariableop_36_dense_6_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37¨
AssignVariableOp_37AssignVariableOp assignvariableop_37_dense_6_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Έ
AssignVariableOp_38AssignVariableOp0assignvariableop_38_batch_normalization_15_gammaIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39·
AssignVariableOp_39AssignVariableOp/assignvariableop_39_batch_normalization_15_betaIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40Ύ
AssignVariableOp_40AssignVariableOp6assignvariableop_40_batch_normalization_15_moving_meanIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41Β
AssignVariableOp_41AssignVariableOp:assignvariableop_41_batch_normalization_15_moving_varianceIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42ͺ
AssignVariableOp_42AssignVariableOp"assignvariableop_42_dense_7_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43¨
AssignVariableOp_43AssignVariableOp assignvariableop_43_dense_7_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44Έ
AssignVariableOp_44AssignVariableOp0assignvariableop_44_batch_normalization_16_gammaIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45·
AssignVariableOp_45AssignVariableOp/assignvariableop_45_batch_normalization_16_betaIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46Ύ
AssignVariableOp_46AssignVariableOp6assignvariableop_46_batch_normalization_16_moving_meanIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47Β
AssignVariableOp_47AssignVariableOp:assignvariableop_47_batch_normalization_16_moving_varianceIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48ͺ
AssignVariableOp_48AssignVariableOp"assignvariableop_48_dense_8_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49¨
AssignVariableOp_49AssignVariableOp assignvariableop_49_dense_8_biasIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50Έ
AssignVariableOp_50AssignVariableOp0assignvariableop_50_batch_normalization_17_gammaIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51·
AssignVariableOp_51AssignVariableOp/assignvariableop_51_batch_normalization_17_betaIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52Ύ
AssignVariableOp_52AssignVariableOp6assignvariableop_52_batch_normalization_17_moving_meanIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53Β
AssignVariableOp_53AssignVariableOp:assignvariableop_53_batch_normalization_17_moving_varianceIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54ͺ
AssignVariableOp_54AssignVariableOp"assignvariableop_54_dense_9_kernelIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55¨
AssignVariableOp_55AssignVariableOp assignvariableop_55_dense_9_biasIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_56₯
AssignVariableOp_56AssignVariableOpassignvariableop_56_adam_iterIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57§
AssignVariableOp_57AssignVariableOpassignvariableop_57_adam_beta_1Identity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58§
AssignVariableOp_58AssignVariableOpassignvariableop_58_adam_beta_2Identity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59¦
AssignVariableOp_59AssignVariableOpassignvariableop_59_adam_decayIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp&assignvariableop_60_adam_learning_rateIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61‘
AssignVariableOp_61AssignVariableOpassignvariableop_61_totalIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62‘
AssignVariableOp_62AssignVariableOpassignvariableop_62_countIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63²
AssignVariableOp_63AssignVariableOp*assignvariableop_63_adam_conv2d_3_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64°
AssignVariableOp_64AssignVariableOp(assignvariableop_64_adam_conv2d_3_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65Ύ
AssignVariableOp_65AssignVariableOp6assignvariableop_65_adam_batch_normalization_9_gamma_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66½
AssignVariableOp_66AssignVariableOp5assignvariableop_66_adam_batch_normalization_9_beta_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67²
AssignVariableOp_67AssignVariableOp*assignvariableop_67_adam_conv2d_4_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68°
AssignVariableOp_68AssignVariableOp(assignvariableop_68_adam_conv2d_4_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69Ώ
AssignVariableOp_69AssignVariableOp7assignvariableop_69_adam_batch_normalization_10_gamma_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70Ύ
AssignVariableOp_70AssignVariableOp6assignvariableop_70_adam_batch_normalization_10_beta_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71²
AssignVariableOp_71AssignVariableOp*assignvariableop_71_adam_conv2d_5_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72°
AssignVariableOp_72AssignVariableOp(assignvariableop_72_adam_conv2d_5_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73Ώ
AssignVariableOp_73AssignVariableOp7assignvariableop_73_adam_batch_normalization_11_gamma_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74Ύ
AssignVariableOp_74AssignVariableOp6assignvariableop_74_adam_batch_normalization_11_beta_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75²
AssignVariableOp_75AssignVariableOp*assignvariableop_75_adam_conv1d_2_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76°
AssignVariableOp_76AssignVariableOp(assignvariableop_76_adam_conv1d_2_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77Ώ
AssignVariableOp_77AssignVariableOp7assignvariableop_77_adam_batch_normalization_12_gamma_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78Ύ
AssignVariableOp_78AssignVariableOp6assignvariableop_78_adam_batch_normalization_12_beta_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79±
AssignVariableOp_79AssignVariableOp)assignvariableop_79_adam_dense_5_kernel_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80―
AssignVariableOp_80AssignVariableOp'assignvariableop_80_adam_dense_5_bias_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81Ώ
AssignVariableOp_81AssignVariableOp7assignvariableop_81_adam_batch_normalization_13_gamma_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82Ύ
AssignVariableOp_82AssignVariableOp6assignvariableop_82_adam_batch_normalization_13_beta_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83²
AssignVariableOp_83AssignVariableOp*assignvariableop_83_adam_conv1d_3_kernel_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84°
AssignVariableOp_84AssignVariableOp(assignvariableop_84_adam_conv1d_3_bias_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85Ώ
AssignVariableOp_85AssignVariableOp7assignvariableop_85_adam_batch_normalization_14_gamma_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86Ύ
AssignVariableOp_86AssignVariableOp6assignvariableop_86_adam_batch_normalization_14_beta_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87±
AssignVariableOp_87AssignVariableOp)assignvariableop_87_adam_dense_6_kernel_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88―
AssignVariableOp_88AssignVariableOp'assignvariableop_88_adam_dense_6_bias_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89Ώ
AssignVariableOp_89AssignVariableOp7assignvariableop_89_adam_batch_normalization_15_gamma_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90Ύ
AssignVariableOp_90AssignVariableOp6assignvariableop_90_adam_batch_normalization_15_beta_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91±
AssignVariableOp_91AssignVariableOp)assignvariableop_91_adam_dense_7_kernel_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92―
AssignVariableOp_92AssignVariableOp'assignvariableop_92_adam_dense_7_bias_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93Ώ
AssignVariableOp_93AssignVariableOp7assignvariableop_93_adam_batch_normalization_16_gamma_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94Ύ
AssignVariableOp_94AssignVariableOp6assignvariableop_94_adam_batch_normalization_16_beta_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95±
AssignVariableOp_95AssignVariableOp)assignvariableop_95_adam_dense_8_kernel_mIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96―
AssignVariableOp_96AssignVariableOp'assignvariableop_96_adam_dense_8_bias_mIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97Ώ
AssignVariableOp_97AssignVariableOp7assignvariableop_97_adam_batch_normalization_17_gamma_mIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98Ύ
AssignVariableOp_98AssignVariableOp6assignvariableop_98_adam_batch_normalization_17_beta_mIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99±
AssignVariableOp_99AssignVariableOp)assignvariableop_99_adam_dense_9_kernel_mIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100³
AssignVariableOp_100AssignVariableOp(assignvariableop_100_adam_dense_9_bias_mIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101Ά
AssignVariableOp_101AssignVariableOp+assignvariableop_101_adam_conv2d_3_kernel_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102΄
AssignVariableOp_102AssignVariableOp)assignvariableop_102_adam_conv2d_3_bias_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103Β
AssignVariableOp_103AssignVariableOp7assignvariableop_103_adam_batch_normalization_9_gamma_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_103q
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:2
Identity_104Α
AssignVariableOp_104AssignVariableOp6assignvariableop_104_adam_batch_normalization_9_beta_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_104q
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:2
Identity_105Ά
AssignVariableOp_105AssignVariableOp+assignvariableop_105_adam_conv2d_4_kernel_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_105q
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:2
Identity_106΄
AssignVariableOp_106AssignVariableOp)assignvariableop_106_adam_conv2d_4_bias_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_106q
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:2
Identity_107Γ
AssignVariableOp_107AssignVariableOp8assignvariableop_107_adam_batch_normalization_10_gamma_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_107q
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:2
Identity_108Β
AssignVariableOp_108AssignVariableOp7assignvariableop_108_adam_batch_normalization_10_beta_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_108q
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:2
Identity_109Ά
AssignVariableOp_109AssignVariableOp+assignvariableop_109_adam_conv2d_5_kernel_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109q
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:2
Identity_110΄
AssignVariableOp_110AssignVariableOp)assignvariableop_110_adam_conv2d_5_bias_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_110q
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:2
Identity_111Γ
AssignVariableOp_111AssignVariableOp8assignvariableop_111_adam_batch_normalization_11_gamma_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_111q
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:2
Identity_112Β
AssignVariableOp_112AssignVariableOp7assignvariableop_112_adam_batch_normalization_11_beta_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_112q
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:2
Identity_113Ά
AssignVariableOp_113AssignVariableOp+assignvariableop_113_adam_conv1d_2_kernel_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_113q
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:2
Identity_114΄
AssignVariableOp_114AssignVariableOp)assignvariableop_114_adam_conv1d_2_bias_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_114q
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:2
Identity_115Γ
AssignVariableOp_115AssignVariableOp8assignvariableop_115_adam_batch_normalization_12_gamma_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_115q
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:2
Identity_116Β
AssignVariableOp_116AssignVariableOp7assignvariableop_116_adam_batch_normalization_12_beta_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_116q
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:2
Identity_117΅
AssignVariableOp_117AssignVariableOp*assignvariableop_117_adam_dense_5_kernel_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_117q
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:2
Identity_118³
AssignVariableOp_118AssignVariableOp(assignvariableop_118_adam_dense_5_bias_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_118q
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:2
Identity_119Γ
AssignVariableOp_119AssignVariableOp8assignvariableop_119_adam_batch_normalization_13_gamma_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119q
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:2
Identity_120Β
AssignVariableOp_120AssignVariableOp7assignvariableop_120_adam_batch_normalization_13_beta_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_120q
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:2
Identity_121Ά
AssignVariableOp_121AssignVariableOp+assignvariableop_121_adam_conv1d_3_kernel_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_121q
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:2
Identity_122΄
AssignVariableOp_122AssignVariableOp)assignvariableop_122_adam_conv1d_3_bias_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_122q
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:2
Identity_123Γ
AssignVariableOp_123AssignVariableOp8assignvariableop_123_adam_batch_normalization_14_gamma_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_123q
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:2
Identity_124Β
AssignVariableOp_124AssignVariableOp7assignvariableop_124_adam_batch_normalization_14_beta_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_124q
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:2
Identity_125΅
AssignVariableOp_125AssignVariableOp*assignvariableop_125_adam_dense_6_kernel_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_125q
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:2
Identity_126³
AssignVariableOp_126AssignVariableOp(assignvariableop_126_adam_dense_6_bias_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_126q
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:2
Identity_127Γ
AssignVariableOp_127AssignVariableOp8assignvariableop_127_adam_batch_normalization_15_gamma_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_127q
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:2
Identity_128Β
AssignVariableOp_128AssignVariableOp7assignvariableop_128_adam_batch_normalization_15_beta_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_128q
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:2
Identity_129΅
AssignVariableOp_129AssignVariableOp*assignvariableop_129_adam_dense_7_kernel_vIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_129q
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:2
Identity_130³
AssignVariableOp_130AssignVariableOp(assignvariableop_130_adam_dense_7_bias_vIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_130q
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:2
Identity_131Γ
AssignVariableOp_131AssignVariableOp8assignvariableop_131_adam_batch_normalization_16_gamma_vIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_131q
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:2
Identity_132Β
AssignVariableOp_132AssignVariableOp7assignvariableop_132_adam_batch_normalization_16_beta_vIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_132q
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:2
Identity_133΅
AssignVariableOp_133AssignVariableOp*assignvariableop_133_adam_dense_8_kernel_vIdentity_133:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_133q
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:2
Identity_134³
AssignVariableOp_134AssignVariableOp(assignvariableop_134_adam_dense_8_bias_vIdentity_134:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_134q
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:2
Identity_135Γ
AssignVariableOp_135AssignVariableOp8assignvariableop_135_adam_batch_normalization_17_gamma_vIdentity_135:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_135q
Identity_136IdentityRestoreV2:tensors:136"/device:CPU:0*
T0*
_output_shapes
:2
Identity_136Β
AssignVariableOp_136AssignVariableOp7assignvariableop_136_adam_batch_normalization_17_beta_vIdentity_136:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_136q
Identity_137IdentityRestoreV2:tensors:137"/device:CPU:0*
T0*
_output_shapes
:2
Identity_137΅
AssignVariableOp_137AssignVariableOp*assignvariableop_137_adam_dense_9_kernel_vIdentity_137:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_137q
Identity_138IdentityRestoreV2:tensors:138"/device:CPU:0*
T0*
_output_shapes
:2
Identity_138³
AssignVariableOp_138AssignVariableOp(assignvariableop_138_adam_dense_9_bias_vIdentity_138:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1389
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpω
Identity_139Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_139ν
Identity_140IdentityIdentity_139:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
T0*
_output_shapes
: 2
Identity_140"%
identity_140Identity_140:output:0*Γ
_input_shapes±
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252,
AssignVariableOp_126AssignVariableOp_1262,
AssignVariableOp_127AssignVariableOp_1272,
AssignVariableOp_128AssignVariableOp_1282,
AssignVariableOp_129AssignVariableOp_1292*
AssignVariableOp_13AssignVariableOp_132,
AssignVariableOp_130AssignVariableOp_1302,
AssignVariableOp_131AssignVariableOp_1312,
AssignVariableOp_132AssignVariableOp_1322,
AssignVariableOp_133AssignVariableOp_1332,
AssignVariableOp_134AssignVariableOp_1342,
AssignVariableOp_135AssignVariableOp_1352,
AssignVariableOp_136AssignVariableOp_1362,
AssignVariableOp_137AssignVariableOp_1372,
AssignVariableOp_138AssignVariableOp_1382*
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
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ι
i
M__inference_tf_op_layer_Mul_1_layer_call_and_return_conditional_losses_403099

inputs
identitys
Mul_1/yConst*
_output_shapes
:*
dtype0*-
value$B""  @@  @@  @@  @@  @@  @@2	
Mul_1/yp
Mul_1MulinputsMul_1/y:output:0*
T0*
_cloned(*'
_output_shapes
:?????????2
Mul_1]
IdentityIdentity	Mul_1:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs

Ή
D__inference_conv1d_2_layer_call_and_return_conditional_losses_402088

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????2
conv1d/ExpandDimsΈ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1·
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:??????????*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:??????????*
squeeze_dims

ύ????????2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:??????????2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:??????????2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :?????????:::T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs


R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_402628

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2
batchnorm/add_1k
IdentityIdentitybatchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????:::::S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs


Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_401728

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ά
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@:::::i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs

―
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_399137

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ή
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:?????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*1
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:?????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:Y U
1
_output_shapes
:?????????
 
_user_specified_nameinputs


Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_397799

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ά
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@:::::i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs

o
S__inference_tf_op_layer_Transpose_1_layer_call_and_return_conditional_losses_399212

inputs
identityy
Transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
Transpose_1/perm
Transpose_1	TransposeinputsTranspose_1/perm:output:0*
T0*
_cloned(*,
_output_shapes
:?????????2
Transpose_1h
IdentityIdentityTranspose_1:y:0*
T0*,
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:?????????:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
Μ
ͺ
7__inference_batch_normalization_15_layer_call_fn_402845

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_3996862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
Ω
-__inference_functional_3_layer_call_fn_401478

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54
identity’StatefulPartitionedCallΦ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54*D
Tin=
;29*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*H
_read_only_resource_inputs*
(&	
 #$%&)*+,/0125678*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_functional_3_layer_call_and_return_conditional_losses_4002072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*
_input_shapesϋ
ψ:?????????::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
Υ)
Λ
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_402419

inputs
assignmovingavg_402394
assignmovingavg_1_402400)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity’#AssignMovingAvg/AssignSubVariableOp’%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradient¨
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:??????????@2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesΆ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze_1
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/402394*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_402394*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpΓ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/402394*
_output_shapes
:@2
AssignMovingAvg/subΊ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/402394*
_output_shapes
:@2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_402394AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/402394*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp€
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/402400*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_402400*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpΝ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/402400*
_output_shapes
:@2
AssignMovingAvg_1/subΔ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/402400*
_output_shapes
:@2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_402400AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/402400*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:??????????@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:??????????@2
batchnorm/add_1Ή
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*+
_output_shapes
:??????????@2

Identity"
identityIdentity:output:0*:
_input_shapes)
':??????????@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:S O
+
_output_shapes
:??????????@
 
_user_specified_nameinputs
Έ)
Λ
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_398689

inputs
assignmovingavg_398664
assignmovingavg_1_398670)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity’#AssignMovingAvg/AssignSubVariableOp’%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	2
moments/StopGradient₯
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:?????????2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices³
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze_1
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/398664*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_398664*
_output_shapes	
:*
dtype02 
AssignMovingAvg/ReadVariableOpΔ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/398664*
_output_shapes	
:2
AssignMovingAvg/sub»
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/398664*
_output_shapes	
:2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_398664AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/398664*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp€
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/398670*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_398670*
_output_shapes	
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpΞ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/398670*
_output_shapes	
:2
AssignMovingAvg_1/subΕ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/398670*
_output_shapes	
:2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_398670AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/398670*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:?????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:?????????2
batchnorm/add_1Ά
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs

―
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_399037

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ω
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:?????????@
 
_user_specified_nameinputs
ώ
L
0__inference_max_pooling1d_1_layer_call_fn_398033

inputs
identityδ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_3980272
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
Ζ

R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_402546

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1t
IdentityIdentitybatchnorm/add_1:z:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????:::::\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
Υ)
Λ
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_399548

inputs
assignmovingavg_399523
assignmovingavg_1_399529)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity’#AssignMovingAvg/AssignSubVariableOp’%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient¨
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:?????????2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesΆ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/399523*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_399523*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpΓ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/399523*
_output_shapes
:2
AssignMovingAvg/subΊ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/399523*
_output_shapes
:2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_399523AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/399523*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp€
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/399529*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_399529*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpΝ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/399529*
_output_shapes
:2
AssignMovingAvg_1/subΔ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/399529*
_output_shapes
:2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_399529AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/399529*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2
batchnorm/add_1Ή
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
Φ
―
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_397976

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ο
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
 )
Λ
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_398829

inputs
assignmovingavg_398804
assignmovingavg_1_398810)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity’#AssignMovingAvg/AssignSubVariableOp’%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@2
moments/StopGradient€
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????@2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices²
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze_1
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/398804*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_398804*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpΓ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/398804*
_output_shapes
:@2
AssignMovingAvg/subΊ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/398804*
_output_shapes
:@2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_398804AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/398804*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp€
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/398810*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_398810*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpΝ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/398810*
_output_shapes
:@2
AssignMovingAvg_1/subΔ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/398810*
_output_shapes
:@2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_398810AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/398810*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????@2
batchnorm/add_1΅
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
Μ
ͺ
7__inference_batch_normalization_12_layer_call_fn_402248

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_3992872
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:??????????
 
_user_specified_nameinputs
Ξ
ͺ
7__inference_batch_normalization_12_layer_call_fn_402261

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall₯
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:??????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_3993072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:??????????
 
_user_specified_nameinputs
σ
~
)__inference_conv1d_2_layer_call_fn_402097

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallύ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_3992362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :?????????::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
*
Λ
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_398129

inputs
assignmovingavg_398104
assignmovingavg_1_398110)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity’#AssignMovingAvg/AssignSubVariableOp’%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesΆ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/398104*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_398104*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpΓ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/398104*
_output_shapes
:2
AssignMovingAvg/subΊ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/398104*
_output_shapes
:2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_398104AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/398104*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp€
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/398110*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_398110*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpΝ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/398110*
_output_shapes
:2
AssignMovingAvg_1/subΔ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/398110*
_output_shapes
:2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_398110AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/398110*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1Β
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
ͺ
ͺ
7__inference_batch_normalization_11_layer_call_fn_401986

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCallΌ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_3980072
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
ο
}
(__inference_dense_6_layer_call_fn_402694

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallό
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_3996352
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
*
Λ
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_402730

inputs
assignmovingavg_402705
assignmovingavg_1_402711)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity’#AssignMovingAvg/AssignSubVariableOp’%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????@2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesΆ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze_1
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/402705*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_402705*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpΓ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/402705*
_output_shapes
:@2
AssignMovingAvg/subΊ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/402705*
_output_shapes
:@2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_402705AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/402705*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp€
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/402711*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_402711*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpΝ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/402711*
_output_shapes
:@2
AssignMovingAvg_1/subΔ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/402711*
_output_shapes
:@2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_402711AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/402711*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????@2
batchnorm/add_1Β
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*4
_output_shapes"
 :??????????????????@2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs
ί
?
C__inference_dense_6_layer_call_and_return_conditional_losses_402685

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisΡ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axisΧ
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????@2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????:::S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
ί
?
C__inference_dense_6_layer_call_and_return_conditional_losses_399635

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisΡ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axisΧ
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????@2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????:::S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
Ζ

R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_398582

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????@2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????@2
batchnorm/add_1t
IdentityIdentitybatchnorm/add_1:z:0*
T0*4
_output_shapes"
 :??????????????????@2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????@:::::\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs
Υ)
Λ
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_399287

inputs
assignmovingavg_399262
assignmovingavg_1_399268)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity’#AssignMovingAvg/AssignSubVariableOp’%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient¨
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:??????????2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesΆ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/399262*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_399262*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpΓ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/399262*
_output_shapes
:2
AssignMovingAvg/subΊ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/399262*
_output_shapes
:2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_399262AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/399262*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp€
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/399268*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_399268*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpΝ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/399268*
_output_shapes
:2
AssignMovingAvg_1/subΔ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/399268*
_output_shapes
:2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_399268AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/399268*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:??????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:??????????2
batchnorm/add_1Ή
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*+
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':??????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:S O
+
_output_shapes
:??????????
 
_user_specified_nameinputs
Υ)
Λ
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_402608

inputs
assignmovingavg_402583
assignmovingavg_1_402589)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity’#AssignMovingAvg/AssignSubVariableOp’%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient¨
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:?????????2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesΆ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/402583*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_402583*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpΓ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/402583*
_output_shapes
:2
AssignMovingAvg/subΊ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/402583*
_output_shapes
:2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_402583AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/402583*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp€
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/402589*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_402589*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpΝ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/402589*
_output_shapes
:2
AssignMovingAvg_1/subΔ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/402589*
_output_shapes
:2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_402589AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/402589*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2
batchnorm/add_1Ή
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
£
F
*__inference_flatten_1_layer_call_fn_402869

inputs
identityΙ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_3997482
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????@:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
’
©
6__inference_batch_normalization_9_layer_call_fn_401741

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCallΈ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_3977682
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
Υ)
Λ
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_399425

inputs
assignmovingavg_399400
assignmovingavg_1_399406)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity’#AssignMovingAvg/AssignSubVariableOp’%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradient¨
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:??????????@2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesΆ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze_1
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/399400*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_399400*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpΓ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/399400*
_output_shapes
:@2
AssignMovingAvg/subΊ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/399400*
_output_shapes
:@2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_399400AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/399400*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp€
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/399406*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_399406*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpΝ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/399406*
_output_shapes
:@2
AssignMovingAvg_1/subΔ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/399406*
_output_shapes
:@2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_399406AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/399406*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:??????????@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:??????????@2
batchnorm/add_1Ή
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*+
_output_shapes
:??????????@2

Identity"
identityIdentity:output:0*:
_input_shapes)
':??????????@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:S O
+
_output_shapes
:??????????@
 
_user_specified_nameinputs
Τ

Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_401664

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Λ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????@:::::X T
0
_output_shapes
:?????????@
 
_user_specified_nameinputs


R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_399307

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:??????????2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:??????????2
batchnorm/add_1k
IdentityIdentitybatchnorm/add_1:z:0*
T0*+
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':??????????:::::S O
+
_output_shapes
:??????????
 
_user_specified_nameinputs
ί
}
(__inference_dense_9_layer_call_fn_403093

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallψ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_3998912
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
Ύ
ͺ
7__inference_batch_normalization_17_layer_call_fn_403073

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall‘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_3988622
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
	
¬
D__inference_conv2d_5_layer_call_and_return_conditional_losses_399102

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¦
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:?????????2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????@:::X T
0
_output_shapes
:?????????@
 
_user_specified_nameinputs
π
ͺ
7__inference_batch_normalization_12_layer_call_fn_402166

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_3981292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs


R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_402832

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:?????????@2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????@2
batchnorm/add_1k
IdentityIdentitybatchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????@:::::S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
©
N
2__inference_tf_op_layer_Mul_1_layer_call_fn_403104

inputs
identityΠ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_tf_op_layer_Mul_1_layer_call_and_return_conditional_losses_3999132
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
Ζ

R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_402750

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????@2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????@2
batchnorm/add_1t
IdentityIdentitybatchnorm/add_1:z:0*
T0*4
_output_shapes"
 :??????????????????@2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????@:::::\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs


R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_403047

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????@2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????@2
batchnorm/add_1g
IdentityIdentitybatchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@:::::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs"ΈL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ή
serving_default₯
@
input_25
serving_default_input_2:0?????????E
tf_op_layer_Mul_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:
₯ρ
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer-9
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer_with_weights-8
layer-13
layer_with_weights-9
layer-14
layer_with_weights-10
layer-15
layer_with_weights-11
layer-16
layer_with_weights-12
layer-17
layer_with_weights-13
layer-18
layer-19
layer_with_weights-14
layer-20
layer_with_weights-15
layer-21
layer_with_weights-16
layer-22
layer_with_weights-17
layer-23
layer_with_weights-18
layer-24
layer-25
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api
 
signatures
+&call_and_return_all_conditional_losses
__call__
 _default_save_signature"ύθ
_tf_keras_networkΰθ{"class_name": "Functional", "name": "functional_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 512, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "ExpandDims_1", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims_1", "op": "ExpandDims", "input": ["input_2", "ExpandDims_1/dim"], "attr": {"T": {"type": "DT_FLOAT"}, "Tdim": {"type": "DT_INT32"}}}, "constants": {"1": -1}}, "name": "tf_op_layer_ExpandDims_1", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["tf_op_layer_ExpandDims_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_9", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["batch_normalization_9", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_10", "inbound_nodes": [[["conv2d_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_5", "inbound_nodes": [[["batch_normalization_10", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_11", "inbound_nodes": [[["conv2d_5", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Reshape_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Reshape_1", "op": "Reshape", "input": ["batch_normalization_11/cond/Identity", "Reshape_1/shape"], "attr": {"T": {"type": "DT_FLOAT"}, "Tshape": {"type": "DT_INT32"}}}, "constants": {"1": [-1, 512, 256]}}, "name": "tf_op_layer_Reshape_1", "inbound_nodes": [[["batch_normalization_11", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [256]}, "pool_size": {"class_name": "__tuple__", "items": [256]}, "padding": "valid", "data_format": "channels_last"}, "name": "max_pooling1d_1", "inbound_nodes": [[["tf_op_layer_Reshape_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Transpose_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Transpose_1", "op": "Transpose", "input": ["max_pooling1d_1/Squeeze", "Transpose_1/perm"], "attr": {"Tperm": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": [0, 2, 1]}}, "name": "tf_op_layer_Transpose_1", "inbound_nodes": [[["max_pooling1d_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [8]}, "strides": {"class_name": "__tuple__", "items": [4]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_2", "inbound_nodes": [[["tf_op_layer_Transpose_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_12", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_12", "inbound_nodes": [[["conv1d_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["batch_normalization_12", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_13", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_13", "inbound_nodes": [[["dense_5", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_3", "trainable": true, "dtype": "float32", "filters": 2, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_3", "inbound_nodes": [[["batch_normalization_13", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_14", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_14", "inbound_nodes": [[["conv1d_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["batch_normalization_14", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_15", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_15", "inbound_nodes": [[["dense_6", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["batch_normalization_15", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_7", "inbound_nodes": [[["flatten_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_16", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_16", "inbound_nodes": [[["dense_7", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_8", "inbound_nodes": [[["batch_normalization_16", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_17", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_17", "inbound_nodes": [[["dense_8", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 6, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_9", "inbound_nodes": [[["batch_normalization_17", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_1", "op": "Mul", "input": ["dense_9/Tanh", "Mul_1/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": [3.0, 3.0, 3.0, 3.0, 3.0, 3.0]}}, "name": "tf_op_layer_Mul_1", "inbound_nodes": [[["dense_9", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["tf_op_layer_Mul_1", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 512, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "ExpandDims_1", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims_1", "op": "ExpandDims", "input": ["input_2", "ExpandDims_1/dim"], "attr": {"T": {"type": "DT_FLOAT"}, "Tdim": {"type": "DT_INT32"}}}, "constants": {"1": -1}}, "name": "tf_op_layer_ExpandDims_1", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["tf_op_layer_ExpandDims_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_9", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["batch_normalization_9", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_10", "inbound_nodes": [[["conv2d_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_5", "inbound_nodes": [[["batch_normalization_10", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_11", "inbound_nodes": [[["conv2d_5", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Reshape_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Reshape_1", "op": "Reshape", "input": ["batch_normalization_11/cond/Identity", "Reshape_1/shape"], "attr": {"T": {"type": "DT_FLOAT"}, "Tshape": {"type": "DT_INT32"}}}, "constants": {"1": [-1, 512, 256]}}, "name": "tf_op_layer_Reshape_1", "inbound_nodes": [[["batch_normalization_11", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [256]}, "pool_size": {"class_name": "__tuple__", "items": [256]}, "padding": "valid", "data_format": "channels_last"}, "name": "max_pooling1d_1", "inbound_nodes": [[["tf_op_layer_Reshape_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Transpose_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Transpose_1", "op": "Transpose", "input": ["max_pooling1d_1/Squeeze", "Transpose_1/perm"], "attr": {"Tperm": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": [0, 2, 1]}}, "name": "tf_op_layer_Transpose_1", "inbound_nodes": [[["max_pooling1d_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [8]}, "strides": {"class_name": "__tuple__", "items": [4]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_2", "inbound_nodes": [[["tf_op_layer_Transpose_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_12", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_12", "inbound_nodes": [[["conv1d_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["batch_normalization_12", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_13", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_13", "inbound_nodes": [[["dense_5", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_3", "trainable": true, "dtype": "float32", "filters": 2, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_3", "inbound_nodes": [[["batch_normalization_13", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_14", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_14", "inbound_nodes": [[["conv1d_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["batch_normalization_14", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_15", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_15", "inbound_nodes": [[["dense_6", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["batch_normalization_15", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_7", "inbound_nodes": [[["flatten_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_16", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_16", "inbound_nodes": [[["dense_7", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_8", "inbound_nodes": [[["batch_normalization_16", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_17", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_17", "inbound_nodes": [[["dense_8", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 6, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_9", "inbound_nodes": [[["batch_normalization_17", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_1", "op": "Mul", "input": ["dense_9/Tanh", "Mul_1/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": [3.0, 3.0, 3.0, 3.0, 3.0, 3.0]}}, "name": "tf_op_layer_Mul_1", "inbound_nodes": [[["dense_9", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["tf_op_layer_Mul_1", 0, 0]]}}, "training_config": {"loss": {"class_name": "MeanAbsoluteError", "config": {"reduction": "auto", "name": "mean_absolute_error"}}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
σ"π
_tf_keras_input_layerΠ{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 512, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 512, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}

!regularization_losses
"trainable_variables
#	variables
$	keras_api
+‘&call_and_return_all_conditional_losses
’__call__"
_tf_keras_layerθ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_ExpandDims_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "ExpandDims_1", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims_1", "op": "ExpandDims", "input": ["input_2", "ExpandDims_1/dim"], "attr": {"T": {"type": "DT_FLOAT"}, "Tdim": {"type": "DT_INT32"}}}, "constants": {"1": -1}}}
σ	

%kernel
&bias
'regularization_losses
(trainable_variables
)	variables
*	keras_api
+£&call_and_return_all_conditional_losses
€__call__"Μ
_tf_keras_layer²{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512, 3, 1]}}
Ό	
+axis
	,gamma
-beta
.moving_mean
/moving_variance
0regularization_losses
1trainable_variables
2	variables
3	keras_api
+₯&call_and_return_all_conditional_losses
¦__call__"ζ
_tf_keras_layerΜ{"class_name": "BatchNormalization", "name": "batch_normalization_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512, 1, 64]}}
υ	

4kernel
5bias
6regularization_losses
7trainable_variables
8	variables
9	keras_api
+§&call_and_return_all_conditional_losses
¨__call__"Ξ
_tf_keras_layer΄{"class_name": "Conv2D", "name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512, 1, 64]}}
Ύ	
:axis
	;gamma
<beta
=moving_mean
>moving_variance
?regularization_losses
@trainable_variables
A	variables
B	keras_api
+©&call_and_return_all_conditional_losses
ͺ__call__"θ
_tf_keras_layerΞ{"class_name": "BatchNormalization", "name": "batch_normalization_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512, 1, 64]}}
φ	

Ckernel
Dbias
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
+«&call_and_return_all_conditional_losses
¬__call__"Ο
_tf_keras_layer΅{"class_name": "Conv2D", "name": "conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512, 1, 64]}}
ΐ	
Iaxis
	Jgamma
Kbeta
Lmoving_mean
Mmoving_variance
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
+­&call_and_return_all_conditional_losses
?__call__"κ
_tf_keras_layerΠ{"class_name": "BatchNormalization", "name": "batch_normalization_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512, 1, 256]}}
±
Rregularization_losses
Strainable_variables
T	variables
U	keras_api
+―&call_and_return_all_conditional_losses
°__call__" 
_tf_keras_layer{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Reshape_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Reshape_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Reshape_1", "op": "Reshape", "input": ["batch_normalization_11/cond/Identity", "Reshape_1/shape"], "attr": {"T": {"type": "DT_FLOAT"}, "Tshape": {"type": "DT_INT32"}}}, "constants": {"1": [-1, 512, 256]}}}
?
Vregularization_losses
Wtrainable_variables
X	variables
Y	keras_api
+±&call_and_return_all_conditional_losses
²__call__"ξ
_tf_keras_layerΤ{"class_name": "MaxPooling1D", "name": "max_pooling1d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling1d_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [256]}, "pool_size": {"class_name": "__tuple__", "items": [256]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
§
Zregularization_losses
[trainable_variables
\	variables
]	keras_api
+³&call_and_return_all_conditional_losses
΄__call__"
_tf_keras_layerό{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Transpose_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Transpose_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Transpose_1", "op": "Transpose", "input": ["max_pooling1d_1/Squeeze", "Transpose_1/perm"], "attr": {"Tperm": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": [0, 2, 1]}}}
ζ	

^kernel
_bias
`regularization_losses
atrainable_variables
b	variables
c	keras_api
+΅&call_and_return_all_conditional_losses
Ά__call__"Ώ
_tf_keras_layer₯{"class_name": "Conv1D", "name": "conv1d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [8]}, "strides": {"class_name": "__tuple__", "items": [4]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 2}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 2]}}
Έ	
daxis
	egamma
fbeta
gmoving_mean
hmoving_variance
iregularization_losses
jtrainable_variables
k	variables
l	keras_api
+·&call_and_return_all_conditional_losses
Έ__call__"β
_tf_keras_layerΘ{"class_name": "BatchNormalization", "name": "batch_normalization_12", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_12", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 63, 4]}}
τ

mkernel
nbias
oregularization_losses
ptrainable_variables
q	variables
r	keras_api
+Ή&call_and_return_all_conditional_losses
Ί__call__"Ν
_tf_keras_layer³{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 63, 4]}}
Ί	
saxis
	tgamma
ubeta
vmoving_mean
wmoving_variance
xregularization_losses
ytrainable_variables
z	variables
{	keras_api
+»&call_and_return_all_conditional_losses
Ό__call__"δ
_tf_keras_layerΚ{"class_name": "BatchNormalization", "name": "batch_normalization_13", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_13", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 63, 64]}}
ι	

|kernel
}bias
~regularization_losses
trainable_variables
	variables
	keras_api
+½&call_and_return_all_conditional_losses
Ύ__call__"ΐ
_tf_keras_layer¦{"class_name": "Conv1D", "name": "conv1d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_3", "trainable": true, "dtype": "float32", "filters": 2, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 63, 64]}}
Α	
	axis

gamma
	beta
moving_mean
moving_variance
regularization_losses
trainable_variables
	variables
	keras_api
+Ώ&call_and_return_all_conditional_losses
ΐ__call__"β
_tf_keras_layerΘ{"class_name": "BatchNormalization", "name": "batch_normalization_14", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_14", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 2}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 2]}}
ϊ
kernel
	bias
regularization_losses
trainable_variables
	variables
	keras_api
+Α&call_and_return_all_conditional_losses
Β__call__"Ν
_tf_keras_layer³{"class_name": "Dense", "name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 2]}}
Γ	
	axis

gamma
	beta
moving_mean
moving_variance
regularization_losses
trainable_variables
	variables
	keras_api
+Γ&call_and_return_all_conditional_losses
Δ__call__"δ
_tf_keras_layerΚ{"class_name": "BatchNormalization", "name": "batch_normalization_15", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_15", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 64]}}
μ
regularization_losses
trainable_variables
	variables
	keras_api
+Ε&call_and_return_all_conditional_losses
Ζ__call__"Χ
_tf_keras_layer½{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ύ
kernel
	bias
 regularization_losses
‘trainable_variables
’	variables
£	keras_api
+Η&call_and_return_all_conditional_losses
Θ__call__"Π
_tf_keras_layerΆ{"class_name": "Dense", "name": "dense_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1920}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1920]}}
Α	
	€axis

₯gamma
	¦beta
§moving_mean
¨moving_variance
©regularization_losses
ͺtrainable_variables
«	variables
¬	keras_api
+Ι&call_and_return_all_conditional_losses
Κ__call__"β
_tf_keras_layerΘ{"class_name": "BatchNormalization", "name": "batch_normalization_16", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_16", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
ϊ
­kernel
	?bias
―regularization_losses
°trainable_variables
±	variables
²	keras_api
+Λ&call_and_return_all_conditional_losses
Μ__call__"Ν
_tf_keras_layer³{"class_name": "Dense", "name": "dense_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
Ώ	
	³axis

΄gamma
	΅beta
Άmoving_mean
·moving_variance
Έregularization_losses
Ήtrainable_variables
Ί	variables
»	keras_api
+Ν&call_and_return_all_conditional_losses
Ξ__call__"ΰ
_tf_keras_layerΖ{"class_name": "BatchNormalization", "name": "batch_normalization_17", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_17", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
χ
Όkernel
	½bias
Ύregularization_losses
Ώtrainable_variables
ΐ	variables
Α	keras_api
+Ο&call_and_return_all_conditional_losses
Π__call__"Κ
_tf_keras_layer°{"class_name": "Dense", "name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 6, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
υ
Βregularization_losses
Γtrainable_variables
Δ	variables
Ε	keras_api
+Ρ&call_and_return_all_conditional_losses
?__call__"ΰ
_tf_keras_layerΖ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Mul_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Mul_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_1", "op": "Mul", "input": ["dense_9/Tanh", "Mul_1/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": [3.0, 3.0, 3.0, 3.0, 3.0, 3.0]}}}
π
	Ζiter
Ηbeta_1
Θbeta_2

Ιdecay
Κlearning_rate%m?&mΣ,mΤ-mΥ4mΦ5mΧ;mΨ<mΩCmΪDmΫJmάKmέ^mή_mίemΰfmαmmβnmγtmδumε|mζ}mη	mθ	mι	mκ	mλ	mμ	mν	mξ	mο	₯mπ	¦mρ	­mς	?mσ	΄mτ	΅mυ	Όmφ	½mχ%vψ&vω,vϊ-vϋ4vό5vύ;vώ<v?CvDvJvKv^v_vevfvmvnvtvuv|v}v	v	v	v	v	v	v	v	v	₯v	¦v	­v	?v	΄v	΅v	Όv	½v"
	optimizer
 "
trackable_list_wrapper
Φ
%0
&1
,2
-3
44
55
;6
<7
C8
D9
J10
K11
^12
_13
e14
f15
m16
n17
t18
u19
|20
}21
22
23
24
25
26
27
28
29
₯30
¦31
­32
?33
΄34
΅35
Ό36
½37"
trackable_list_wrapper
ξ
%0
&1
,2
-3
.4
/5
46
57
;8
<9
=10
>11
C12
D13
J14
K15
L16
M17
^18
_19
e20
f21
g22
h23
m24
n25
t26
u27
v28
w29
|30
}31
32
33
34
35
36
37
38
39
40
41
42
43
₯44
¦45
§46
¨47
­48
?49
΄50
΅51
Ά52
·53
Ό54
½55"
trackable_list_wrapper
Σ
Λlayer_metrics
 Μlayer_regularization_losses
Νlayers
Ξnon_trainable_variables
regularization_losses
trainable_variables
	variables
Οmetrics
__call__
 _default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
Σserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
Πlayer_metrics
 Ρlayer_regularization_losses
?layers
Σnon_trainable_variables
!regularization_losses
"trainable_variables
#	variables
Τmetrics
’__call__
+‘&call_and_return_all_conditional_losses
'‘"call_and_return_conditional_losses"
_generic_user_object
):'@2conv2d_3/kernel
:@2conv2d_3/bias
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
΅
Υlayer_metrics
 Φlayer_regularization_losses
Χlayers
Ψnon_trainable_variables
'regularization_losses
(trainable_variables
)	variables
Ωmetrics
€__call__
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_9/gamma
(:&@2batch_normalization_9/beta
1:/@ (2!batch_normalization_9/moving_mean
5:3@ (2%batch_normalization_9/moving_variance
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
<
,0
-1
.2
/3"
trackable_list_wrapper
΅
Ϊlayer_metrics
 Ϋlayer_regularization_losses
άlayers
έnon_trainable_variables
0regularization_losses
1trainable_variables
2	variables
ήmetrics
¦__call__
+₯&call_and_return_all_conditional_losses
'₯"call_and_return_conditional_losses"
_generic_user_object
):'@@2conv2d_4/kernel
:@2conv2d_4/bias
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
΅
ίlayer_metrics
 ΰlayer_regularization_losses
αlayers
βnon_trainable_variables
6regularization_losses
7trainable_variables
8	variables
γmetrics
¨__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_10/gamma
):'@2batch_normalization_10/beta
2:0@ (2"batch_normalization_10/moving_mean
6:4@ (2&batch_normalization_10/moving_variance
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
<
;0
<1
=2
>3"
trackable_list_wrapper
΅
δlayer_metrics
 εlayer_regularization_losses
ζlayers
ηnon_trainable_variables
?regularization_losses
@trainable_variables
A	variables
θmetrics
ͺ__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses"
_generic_user_object
*:(@2conv2d_5/kernel
:2conv2d_5/bias
 "
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
΅
ιlayer_metrics
 κlayer_regularization_losses
λlayers
μnon_trainable_variables
Eregularization_losses
Ftrainable_variables
G	variables
νmetrics
¬__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)2batch_normalization_11/gamma
*:(2batch_normalization_11/beta
3:1 (2"batch_normalization_11/moving_mean
7:5 (2&batch_normalization_11/moving_variance
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
<
J0
K1
L2
M3"
trackable_list_wrapper
΅
ξlayer_metrics
 οlayer_regularization_losses
πlayers
ρnon_trainable_variables
Nregularization_losses
Otrainable_variables
P	variables
ςmetrics
?__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
σlayer_metrics
 τlayer_regularization_losses
υlayers
φnon_trainable_variables
Rregularization_losses
Strainable_variables
T	variables
χmetrics
°__call__
+―&call_and_return_all_conditional_losses
'―"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
ψlayer_metrics
 ωlayer_regularization_losses
ϊlayers
ϋnon_trainable_variables
Vregularization_losses
Wtrainable_variables
X	variables
όmetrics
²__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
ύlayer_metrics
 ώlayer_regularization_losses
?layers
non_trainable_variables
Zregularization_losses
[trainable_variables
\	variables
metrics
΄__call__
+³&call_and_return_all_conditional_losses
'³"call_and_return_conditional_losses"
_generic_user_object
%:#2conv1d_2/kernel
:2conv1d_2/bias
 "
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
΅
layer_metrics
 layer_regularization_losses
layers
non_trainable_variables
`regularization_losses
atrainable_variables
b	variables
metrics
Ά__call__
+΅&call_and_return_all_conditional_losses
'΅"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_12/gamma
):'2batch_normalization_12/beta
2:0 (2"batch_normalization_12/moving_mean
6:4 (2&batch_normalization_12/moving_variance
 "
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
<
e0
f1
g2
h3"
trackable_list_wrapper
΅
layer_metrics
 layer_regularization_losses
layers
non_trainable_variables
iregularization_losses
jtrainable_variables
k	variables
metrics
Έ__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
 :@2dense_5/kernel
:@2dense_5/bias
 "
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
΅
layer_metrics
 layer_regularization_losses
layers
non_trainable_variables
oregularization_losses
ptrainable_variables
q	variables
metrics
Ί__call__
+Ή&call_and_return_all_conditional_losses
'Ή"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_13/gamma
):'@2batch_normalization_13/beta
2:0@ (2"batch_normalization_13/moving_mean
6:4@ (2&batch_normalization_13/moving_variance
 "
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
<
t0
u1
v2
w3"
trackable_list_wrapper
΅
layer_metrics
 layer_regularization_losses
layers
non_trainable_variables
xregularization_losses
ytrainable_variables
z	variables
metrics
Ό__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
%:#@2conv1d_3/kernel
:2conv1d_3/bias
 "
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
Ά
layer_metrics
 layer_regularization_losses
layers
non_trainable_variables
~regularization_losses
trainable_variables
	variables
metrics
Ύ__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_14/gamma
):'2batch_normalization_14/beta
2:0 (2"batch_normalization_14/moving_mean
6:4 (2&batch_normalization_14/moving_variance
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
@
0
1
2
3"
trackable_list_wrapper
Έ
layer_metrics
 layer_regularization_losses
layers
non_trainable_variables
regularization_losses
trainable_variables
	variables
metrics
ΐ__call__
+Ώ&call_and_return_all_conditional_losses
'Ώ"call_and_return_conditional_losses"
_generic_user_object
 :@2dense_6/kernel
:@2dense_6/bias
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
Έ
 layer_metrics
 ‘layer_regularization_losses
’layers
£non_trainable_variables
regularization_losses
trainable_variables
	variables
€metrics
Β__call__
+Α&call_and_return_all_conditional_losses
'Α"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_15/gamma
):'@2batch_normalization_15/beta
2:0@ (2"batch_normalization_15/moving_mean
6:4@ (2&batch_normalization_15/moving_variance
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
@
0
1
2
3"
trackable_list_wrapper
Έ
₯layer_metrics
 ¦layer_regularization_losses
§layers
¨non_trainable_variables
regularization_losses
trainable_variables
	variables
©metrics
Δ__call__
+Γ&call_and_return_all_conditional_losses
'Γ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
ͺlayer_metrics
 «layer_regularization_losses
¬layers
­non_trainable_variables
regularization_losses
trainable_variables
	variables
?metrics
Ζ__call__
+Ε&call_and_return_all_conditional_losses
'Ε"call_and_return_conditional_losses"
_generic_user_object
": 
2dense_7/kernel
:2dense_7/bias
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
Έ
―layer_metrics
 °layer_regularization_losses
±layers
²non_trainable_variables
 regularization_losses
‘trainable_variables
’	variables
³metrics
Θ__call__
+Η&call_and_return_all_conditional_losses
'Η"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)2batch_normalization_16/gamma
*:(2batch_normalization_16/beta
3:1 (2"batch_normalization_16/moving_mean
7:5 (2&batch_normalization_16/moving_variance
 "
trackable_list_wrapper
0
₯0
¦1"
trackable_list_wrapper
@
₯0
¦1
§2
¨3"
trackable_list_wrapper
Έ
΄layer_metrics
 ΅layer_regularization_losses
Άlayers
·non_trainable_variables
©regularization_losses
ͺtrainable_variables
«	variables
Έmetrics
Κ__call__
+Ι&call_and_return_all_conditional_losses
'Ι"call_and_return_conditional_losses"
_generic_user_object
!:	@2dense_8/kernel
:@2dense_8/bias
 "
trackable_list_wrapper
0
­0
?1"
trackable_list_wrapper
0
­0
?1"
trackable_list_wrapper
Έ
Ήlayer_metrics
 Ίlayer_regularization_losses
»layers
Όnon_trainable_variables
―regularization_losses
°trainable_variables
±	variables
½metrics
Μ__call__
+Λ&call_and_return_all_conditional_losses
'Λ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_17/gamma
):'@2batch_normalization_17/beta
2:0@ (2"batch_normalization_17/moving_mean
6:4@ (2&batch_normalization_17/moving_variance
 "
trackable_list_wrapper
0
΄0
΅1"
trackable_list_wrapper
@
΄0
΅1
Ά2
·3"
trackable_list_wrapper
Έ
Ύlayer_metrics
 Ώlayer_regularization_losses
ΐlayers
Αnon_trainable_variables
Έregularization_losses
Ήtrainable_variables
Ί	variables
Βmetrics
Ξ__call__
+Ν&call_and_return_all_conditional_losses
'Ν"call_and_return_conditional_losses"
_generic_user_object
 :@2dense_9/kernel
:2dense_9/bias
 "
trackable_list_wrapper
0
Ό0
½1"
trackable_list_wrapper
0
Ό0
½1"
trackable_list_wrapper
Έ
Γlayer_metrics
 Δlayer_regularization_losses
Εlayers
Ζnon_trainable_variables
Ύregularization_losses
Ώtrainable_variables
ΐ	variables
Ηmetrics
Π__call__
+Ο&call_and_return_all_conditional_losses
'Ο"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Θlayer_metrics
 Ιlayer_regularization_losses
Κlayers
Λnon_trainable_variables
Βregularization_losses
Γtrainable_variables
Δ	variables
Μmetrics
?__call__
+Ρ&call_and_return_all_conditional_losses
'Ρ"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
ζ
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
?
.0
/1
=2
>3
L4
M5
g6
h7
v8
w9
10
11
12
13
§14
¨15
Ά16
·17"
trackable_list_wrapper
(
Ν0"
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
v0
w1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
§0
¨1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ά0
·1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ώ

Ξtotal

Οcount
Π	variables
Ρ	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
0
Ξ0
Ο1"
trackable_list_wrapper
.
Π	variables"
_generic_user_object
.:,@2Adam/conv2d_3/kernel/m
 :@2Adam/conv2d_3/bias/m
.:,@2"Adam/batch_normalization_9/gamma/m
-:+@2!Adam/batch_normalization_9/beta/m
.:,@@2Adam/conv2d_4/kernel/m
 :@2Adam/conv2d_4/bias/m
/:-@2#Adam/batch_normalization_10/gamma/m
.:,@2"Adam/batch_normalization_10/beta/m
/:-@2Adam/conv2d_5/kernel/m
!:2Adam/conv2d_5/bias/m
0:.2#Adam/batch_normalization_11/gamma/m
/:-2"Adam/batch_normalization_11/beta/m
*:(2Adam/conv1d_2/kernel/m
 :2Adam/conv1d_2/bias/m
/:-2#Adam/batch_normalization_12/gamma/m
.:,2"Adam/batch_normalization_12/beta/m
%:#@2Adam/dense_5/kernel/m
:@2Adam/dense_5/bias/m
/:-@2#Adam/batch_normalization_13/gamma/m
.:,@2"Adam/batch_normalization_13/beta/m
*:(@2Adam/conv1d_3/kernel/m
 :2Adam/conv1d_3/bias/m
/:-2#Adam/batch_normalization_14/gamma/m
.:,2"Adam/batch_normalization_14/beta/m
%:#@2Adam/dense_6/kernel/m
:@2Adam/dense_6/bias/m
/:-@2#Adam/batch_normalization_15/gamma/m
.:,@2"Adam/batch_normalization_15/beta/m
':%
2Adam/dense_7/kernel/m
 :2Adam/dense_7/bias/m
0:.2#Adam/batch_normalization_16/gamma/m
/:-2"Adam/batch_normalization_16/beta/m
&:$	@2Adam/dense_8/kernel/m
:@2Adam/dense_8/bias/m
/:-@2#Adam/batch_normalization_17/gamma/m
.:,@2"Adam/batch_normalization_17/beta/m
%:#@2Adam/dense_9/kernel/m
:2Adam/dense_9/bias/m
.:,@2Adam/conv2d_3/kernel/v
 :@2Adam/conv2d_3/bias/v
.:,@2"Adam/batch_normalization_9/gamma/v
-:+@2!Adam/batch_normalization_9/beta/v
.:,@@2Adam/conv2d_4/kernel/v
 :@2Adam/conv2d_4/bias/v
/:-@2#Adam/batch_normalization_10/gamma/v
.:,@2"Adam/batch_normalization_10/beta/v
/:-@2Adam/conv2d_5/kernel/v
!:2Adam/conv2d_5/bias/v
0:.2#Adam/batch_normalization_11/gamma/v
/:-2"Adam/batch_normalization_11/beta/v
*:(2Adam/conv1d_2/kernel/v
 :2Adam/conv1d_2/bias/v
/:-2#Adam/batch_normalization_12/gamma/v
.:,2"Adam/batch_normalization_12/beta/v
%:#@2Adam/dense_5/kernel/v
:@2Adam/dense_5/bias/v
/:-@2#Adam/batch_normalization_13/gamma/v
.:,@2"Adam/batch_normalization_13/beta/v
*:(@2Adam/conv1d_3/kernel/v
 :2Adam/conv1d_3/bias/v
/:-2#Adam/batch_normalization_14/gamma/v
.:,2"Adam/batch_normalization_14/beta/v
%:#@2Adam/dense_6/kernel/v
:@2Adam/dense_6/bias/v
/:-@2#Adam/batch_normalization_15/gamma/v
.:,@2"Adam/batch_normalization_15/beta/v
':%
2Adam/dense_7/kernel/v
 :2Adam/dense_7/bias/v
0:.2#Adam/batch_normalization_16/gamma/v
/:-2"Adam/batch_normalization_16/beta/v
&:$	@2Adam/dense_8/kernel/v
:@2Adam/dense_8/bias/v
/:-@2#Adam/batch_normalization_17/gamma/v
.:,@2"Adam/batch_normalization_17/beta/v
%:#@2Adam/dense_9/kernel/v
:2Adam/dense_9/bias/v
ξ2λ
H__inference_functional_3_layer_call_and_return_conditional_losses_400063
H__inference_functional_3_layer_call_and_return_conditional_losses_401085
H__inference_functional_3_layer_call_and_return_conditional_losses_399922
H__inference_functional_3_layer_call_and_return_conditional_losses_401361ΐ
·²³
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
kwonlydefaultsͺ 
annotationsͺ *
 
2?
-__inference_functional_3_layer_call_fn_400580
-__inference_functional_3_layer_call_fn_401478
-__inference_functional_3_layer_call_fn_400322
-__inference_functional_3_layer_call_fn_401595ΐ
·²³
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
kwonlydefaultsͺ 
annotationsͺ *
 
δ2α
!__inference__wrapped_model_397706»
²
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
annotationsͺ *+’(
&#
input_2?????????
ώ2ϋ
T__inference_tf_op_layer_ExpandDims_1_layer_call_and_return_conditional_losses_401601’
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
annotationsͺ *
 
γ2ΰ
9__inference_tf_op_layer_ExpandDims_1_layer_call_fn_401606’
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
annotationsͺ *
 
ξ2λ
D__inference_conv2d_3_layer_call_and_return_conditional_losses_401617’
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
annotationsͺ *
 
Σ2Π
)__inference_conv2d_3_layer_call_fn_401626’
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
annotationsͺ *
 
2
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_401710
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_401664
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_401646
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_401728΄
«²§
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
kwonlydefaultsͺ 
annotationsͺ *
 
2
6__inference_batch_normalization_9_layer_call_fn_401677
6__inference_batch_normalization_9_layer_call_fn_401690
6__inference_batch_normalization_9_layer_call_fn_401741
6__inference_batch_normalization_9_layer_call_fn_401754΄
«²§
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
kwonlydefaultsͺ 
annotationsͺ *
 
ξ2λ
D__inference_conv2d_4_layer_call_and_return_conditional_losses_401765’
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
annotationsͺ *
 
Σ2Π
)__inference_conv2d_4_layer_call_fn_401774’
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
annotationsͺ *
 
2
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_401794
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_401858
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_401812
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_401876΄
«²§
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
kwonlydefaultsͺ 
annotationsͺ *
 
2
7__inference_batch_normalization_10_layer_call_fn_401838
7__inference_batch_normalization_10_layer_call_fn_401825
7__inference_batch_normalization_10_layer_call_fn_401902
7__inference_batch_normalization_10_layer_call_fn_401889΄
«²§
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
kwonlydefaultsͺ 
annotationsͺ *
 
ξ2λ
D__inference_conv2d_5_layer_call_and_return_conditional_losses_401913’
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
annotationsͺ *
 
Σ2Π
)__inference_conv2d_5_layer_call_fn_401922’
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
annotationsͺ *
 
2
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_401942
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_401960
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_402006
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_402024΄
«²§
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
kwonlydefaultsͺ 
annotationsͺ *
 
2
7__inference_batch_normalization_11_layer_call_fn_401986
7__inference_batch_normalization_11_layer_call_fn_402037
7__inference_batch_normalization_11_layer_call_fn_402050
7__inference_batch_normalization_11_layer_call_fn_401973΄
«²§
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
kwonlydefaultsͺ 
annotationsͺ *
 
ϋ2ψ
Q__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_402056’
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
annotationsͺ *
 
ΰ2έ
6__inference_tf_op_layer_Reshape_1_layer_call_fn_402061’
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
annotationsͺ *
 
¦2£
K__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_398027Σ
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
annotationsͺ *3’0
.+'???????????????????????????
2
0__inference_max_pooling1d_1_layer_call_fn_398033Σ
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
annotationsͺ *3’0
.+'???????????????????????????
ύ2ϊ
S__inference_tf_op_layer_Transpose_1_layer_call_and_return_conditional_losses_402067’
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
annotationsͺ *
 
β2ί
8__inference_tf_op_layer_Transpose_1_layer_call_fn_402072’
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
annotationsͺ *
 
ξ2λ
D__inference_conv1d_2_layer_call_and_return_conditional_losses_402088’
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
annotationsͺ *
 
Σ2Π
)__inference_conv1d_2_layer_call_fn_402097’
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
annotationsͺ *
 
2
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_402153
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_402133
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_402215
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_402235΄
«²§
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
kwonlydefaultsͺ 
annotationsͺ *
 
2
7__inference_batch_normalization_12_layer_call_fn_402179
7__inference_batch_normalization_12_layer_call_fn_402248
7__inference_batch_normalization_12_layer_call_fn_402261
7__inference_batch_normalization_12_layer_call_fn_402166΄
«²§
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
kwonlydefaultsͺ 
annotationsͺ *
 
ν2κ
C__inference_dense_5_layer_call_and_return_conditional_losses_402292’
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
annotationsͺ *
 
?2Ο
(__inference_dense_5_layer_call_fn_402301’
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
annotationsͺ *
 
2
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_402357
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_402419
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_402337
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_402439΄
«²§
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
kwonlydefaultsͺ 
annotationsͺ *
 
2
7__inference_batch_normalization_13_layer_call_fn_402383
7__inference_batch_normalization_13_layer_call_fn_402465
7__inference_batch_normalization_13_layer_call_fn_402452
7__inference_batch_normalization_13_layer_call_fn_402370΄
«²§
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
kwonlydefaultsͺ 
annotationsͺ *
 
ξ2λ
D__inference_conv1d_3_layer_call_and_return_conditional_losses_402481’
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
annotationsͺ *
 
Σ2Π
)__inference_conv1d_3_layer_call_fn_402490’
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
annotationsͺ *
 
2
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_402546
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_402608
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_402628
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_402526΄
«²§
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
kwonlydefaultsͺ 
annotationsͺ *
 
2
7__inference_batch_normalization_14_layer_call_fn_402641
7__inference_batch_normalization_14_layer_call_fn_402654
7__inference_batch_normalization_14_layer_call_fn_402572
7__inference_batch_normalization_14_layer_call_fn_402559΄
«²§
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
kwonlydefaultsͺ 
annotationsͺ *
 
ν2κ
C__inference_dense_6_layer_call_and_return_conditional_losses_402685’
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
annotationsͺ *
 
?2Ο
(__inference_dense_6_layer_call_fn_402694’
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
annotationsͺ *
 
2
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_402750
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_402812
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_402730
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_402832΄
«²§
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
kwonlydefaultsͺ 
annotationsͺ *
 
2
7__inference_batch_normalization_15_layer_call_fn_402858
7__inference_batch_normalization_15_layer_call_fn_402776
7__inference_batch_normalization_15_layer_call_fn_402763
7__inference_batch_normalization_15_layer_call_fn_402845΄
«²§
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
kwonlydefaultsͺ 
annotationsͺ *
 
ο2μ
E__inference_flatten_1_layer_call_and_return_conditional_losses_402864’
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
annotationsͺ *
 
Τ2Ρ
*__inference_flatten_1_layer_call_fn_402869’
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
annotationsͺ *
 
ν2κ
C__inference_dense_7_layer_call_and_return_conditional_losses_402880’
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
annotationsͺ *
 
?2Ο
(__inference_dense_7_layer_call_fn_402889’
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
annotationsͺ *
 
β2ί
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_402945
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_402925΄
«²§
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
kwonlydefaultsͺ 
annotationsͺ *
 
¬2©
7__inference_batch_normalization_16_layer_call_fn_402971
7__inference_batch_normalization_16_layer_call_fn_402958΄
«²§
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
kwonlydefaultsͺ 
annotationsͺ *
 
ν2κ
C__inference_dense_8_layer_call_and_return_conditional_losses_402982’
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
annotationsͺ *
 
?2Ο
(__inference_dense_8_layer_call_fn_402991’
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
annotationsͺ *
 
β2ί
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_403047
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_403027΄
«²§
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
kwonlydefaultsͺ 
annotationsͺ *
 
¬2©
7__inference_batch_normalization_17_layer_call_fn_403073
7__inference_batch_normalization_17_layer_call_fn_403060΄
«²§
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
kwonlydefaultsͺ 
annotationsͺ *
 
ν2κ
C__inference_dense_9_layer_call_and_return_conditional_losses_403084’
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
annotationsͺ *
 
?2Ο
(__inference_dense_9_layer_call_fn_403093’
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
annotationsͺ *
 
χ2τ
M__inference_tf_op_layer_Mul_1_layer_call_and_return_conditional_losses_403099’
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
annotationsͺ *
 
ά2Ω
2__inference_tf_op_layer_Mul_1_layer_call_fn_403104’
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
annotationsͺ *
 
3B1
$__inference_signature_wrapper_400707input_2φ
!__inference__wrapped_model_397706ΠP%&,-./45;<=>CDJKLM^_hegfmnwtvu|}¨₯§¦­?·΄Ά΅Ό½5’2
+’(
&#
input_2?????????
ͺ "EͺB
@
tf_op_layer_Mul_1+(
tf_op_layer_Mul_1?????????ν
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_401794;<=>M’J
C’@
:7
inputs+???????????????????????????@
p
ͺ "?’<
52
0+???????????????????????????@
 ν
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_401812;<=>M’J
C’@
:7
inputs+???????????????????????????@
p 
ͺ "?’<
52
0+???????????????????????????@
 Κ
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_401858t;<=><’9
2’/
)&
inputs?????????@
p
ͺ ".’+
$!
0?????????@
 Κ
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_401876t;<=><’9
2’/
)&
inputs?????????@
p 
ͺ ".’+
$!
0?????????@
 Ε
7__inference_batch_normalization_10_layer_call_fn_401825;<=>M’J
C’@
:7
inputs+???????????????????????????@
p
ͺ "2/+???????????????????????????@Ε
7__inference_batch_normalization_10_layer_call_fn_401838;<=>M’J
C’@
:7
inputs+???????????????????????????@
p 
ͺ "2/+???????????????????????????@’
7__inference_batch_normalization_10_layer_call_fn_401889g;<=><’9
2’/
)&
inputs?????????@
p
ͺ "!?????????@’
7__inference_batch_normalization_10_layer_call_fn_401902g;<=><’9
2’/
)&
inputs?????????@
p 
ͺ "!?????????@ο
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_401942JKLMN’K
D’A
;8
inputs,???????????????????????????
p
ͺ "@’=
63
0,???????????????????????????
 ο
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_401960JKLMN’K
D’A
;8
inputs,???????????????????????????
p 
ͺ "@’=
63
0,???????????????????????????
 Μ
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_402006vJKLM=’:
3’0
*'
inputs?????????
p
ͺ "/’,
%"
0?????????
 Μ
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_402024vJKLM=’:
3’0
*'
inputs?????????
p 
ͺ "/’,
%"
0?????????
 Η
7__inference_batch_normalization_11_layer_call_fn_401973JKLMN’K
D’A
;8
inputs,???????????????????????????
p
ͺ "30,???????????????????????????Η
7__inference_batch_normalization_11_layer_call_fn_401986JKLMN’K
D’A
;8
inputs,???????????????????????????
p 
ͺ "30,???????????????????????????€
7__inference_batch_normalization_11_layer_call_fn_402037iJKLM=’:
3’0
*'
inputs?????????
p
ͺ ""?????????€
7__inference_batch_normalization_11_layer_call_fn_402050iJKLM=’:
3’0
*'
inputs?????????
p 
ͺ ""??????????
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_402133|ghef@’=
6’3
-*
inputs??????????????????
p
ͺ "2’/
(%
0??????????????????
 ?
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_402153|hegf@’=
6’3
-*
inputs??????????????????
p 
ͺ "2’/
(%
0??????????????????
 ΐ
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_402215jghef7’4
-’*
$!
inputs??????????
p
ͺ ")’&

0??????????
 ΐ
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_402235jhegf7’4
-’*
$!
inputs??????????
p 
ͺ ")’&

0??????????
 ͺ
7__inference_batch_normalization_12_layer_call_fn_402166oghef@’=
6’3
-*
inputs??????????????????
p
ͺ "%"??????????????????ͺ
7__inference_batch_normalization_12_layer_call_fn_402179ohegf@’=
6’3
-*
inputs??????????????????
p 
ͺ "%"??????????????????
7__inference_batch_normalization_12_layer_call_fn_402248]ghef7’4
-’*
$!
inputs??????????
p
ͺ "??????????
7__inference_batch_normalization_12_layer_call_fn_402261]hegf7’4
-’*
$!
inputs??????????
p 
ͺ "???????????
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_402337|vwtu@’=
6’3
-*
inputs??????????????????@
p
ͺ "2’/
(%
0??????????????????@
 ?
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_402357|wtvu@’=
6’3
-*
inputs??????????????????@
p 
ͺ "2’/
(%
0??????????????????@
 ΐ
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_402419jvwtu7’4
-’*
$!
inputs??????????@
p
ͺ ")’&

0??????????@
 ΐ
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_402439jwtvu7’4
-’*
$!
inputs??????????@
p 
ͺ ")’&

0??????????@
 ͺ
7__inference_batch_normalization_13_layer_call_fn_402370ovwtu@’=
6’3
-*
inputs??????????????????@
p
ͺ "%"??????????????????@ͺ
7__inference_batch_normalization_13_layer_call_fn_402383owtvu@’=
6’3
-*
inputs??????????????????@
p 
ͺ "%"??????????????????@
7__inference_batch_normalization_13_layer_call_fn_402452]vwtu7’4
-’*
$!
inputs??????????@
p
ͺ "??????????@
7__inference_batch_normalization_13_layer_call_fn_402465]wtvu7’4
-’*
$!
inputs??????????@
p 
ͺ "??????????@Χ
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_402526@’=
6’3
-*
inputs??????????????????
p
ͺ "2’/
(%
0??????????????????
 Χ
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_402546@’=
6’3
-*
inputs??????????????????
p 
ͺ "2’/
(%
0??????????????????
 Δ
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_402608n7’4
-’*
$!
inputs?????????
p
ͺ ")’&

0?????????
 Δ
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_402628n7’4
-’*
$!
inputs?????????
p 
ͺ ")’&

0?????????
 ?
7__inference_batch_normalization_14_layer_call_fn_402559s@’=
6’3
-*
inputs??????????????????
p
ͺ "%"???????????????????
7__inference_batch_normalization_14_layer_call_fn_402572s@’=
6’3
-*
inputs??????????????????
p 
ͺ "%"??????????????????
7__inference_batch_normalization_14_layer_call_fn_402641a7’4
-’*
$!
inputs?????????
p
ͺ "?????????
7__inference_batch_normalization_14_layer_call_fn_402654a7’4
-’*
$!
inputs?????????
p 
ͺ "?????????Χ
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_402730@’=
6’3
-*
inputs??????????????????@
p
ͺ "2’/
(%
0??????????????????@
 Χ
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_402750@’=
6’3
-*
inputs??????????????????@
p 
ͺ "2’/
(%
0??????????????????@
 Δ
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_402812n7’4
-’*
$!
inputs?????????@
p
ͺ ")’&

0?????????@
 Δ
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_402832n7’4
-’*
$!
inputs?????????@
p 
ͺ ")’&

0?????????@
 ?
7__inference_batch_normalization_15_layer_call_fn_402763s@’=
6’3
-*
inputs??????????????????@
p
ͺ "%"??????????????????@?
7__inference_batch_normalization_15_layer_call_fn_402776s@’=
6’3
-*
inputs??????????????????@
p 
ͺ "%"??????????????????@
7__inference_batch_normalization_15_layer_call_fn_402845a7’4
-’*
$!
inputs?????????@
p
ͺ "?????????@
7__inference_batch_normalization_15_layer_call_fn_402858a7’4
-’*
$!
inputs?????????@
p 
ͺ "?????????@Ύ
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_402925h§¨₯¦4’1
*’'
!
inputs?????????
p
ͺ "&’#

0?????????
 Ύ
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_402945h¨₯§¦4’1
*’'
!
inputs?????????
p 
ͺ "&’#

0?????????
 
7__inference_batch_normalization_16_layer_call_fn_402958[§¨₯¦4’1
*’'
!
inputs?????????
p
ͺ "?????????
7__inference_batch_normalization_16_layer_call_fn_402971[¨₯§¦4’1
*’'
!
inputs?????????
p 
ͺ "?????????Ό
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_403027fΆ·΄΅3’0
)’&
 
inputs?????????@
p
ͺ "%’"

0?????????@
 Ό
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_403047f·΄Ά΅3’0
)’&
 
inputs?????????@
p 
ͺ "%’"

0?????????@
 
7__inference_batch_normalization_17_layer_call_fn_403060YΆ·΄΅3’0
)’&
 
inputs?????????@
p
ͺ "?????????@
7__inference_batch_normalization_17_layer_call_fn_403073Y·΄Ά΅3’0
)’&
 
inputs?????????@
p 
ͺ "?????????@Ι
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_401646t,-./<’9
2’/
)&
inputs?????????@
p
ͺ ".’+
$!
0?????????@
 Ι
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_401664t,-./<’9
2’/
)&
inputs?????????@
p 
ͺ ".’+
$!
0?????????@
 μ
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_401710,-./M’J
C’@
:7
inputs+???????????????????????????@
p
ͺ "?’<
52
0+???????????????????????????@
 μ
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_401728,-./M’J
C’@
:7
inputs+???????????????????????????@
p 
ͺ "?’<
52
0+???????????????????????????@
 ‘
6__inference_batch_normalization_9_layer_call_fn_401677g,-./<’9
2’/
)&
inputs?????????@
p
ͺ "!?????????@‘
6__inference_batch_normalization_9_layer_call_fn_401690g,-./<’9
2’/
)&
inputs?????????@
p 
ͺ "!?????????@Δ
6__inference_batch_normalization_9_layer_call_fn_401741,-./M’J
C’@
:7
inputs+???????????????????????????@
p
ͺ "2/+???????????????????????????@Δ
6__inference_batch_normalization_9_layer_call_fn_401754,-./M’J
C’@
:7
inputs+???????????????????????????@
p 
ͺ "2/+???????????????????????????@­
D__inference_conv1d_2_layer_call_and_return_conditional_losses_402088e^_4’1
*’'
%"
inputs?????????
ͺ ")’&

0??????????
 
)__inference_conv1d_2_layer_call_fn_402097X^_4’1
*’'
%"
inputs?????????
ͺ "??????????¬
D__inference_conv1d_3_layer_call_and_return_conditional_losses_402481d|}3’0
)’&
$!
inputs??????????@
ͺ ")’&

0?????????
 
)__inference_conv1d_3_layer_call_fn_402490W|}3’0
)’&
$!
inputs??????????@
ͺ "?????????Ά
D__inference_conv2d_3_layer_call_and_return_conditional_losses_401617n%&8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????@
 
)__inference_conv2d_3_layer_call_fn_401626a%&8’5
.’+
)&
inputs?????????
ͺ "!?????????@Ά
D__inference_conv2d_4_layer_call_and_return_conditional_losses_401765n458’5
.’+
)&
inputs?????????@
ͺ ".’+
$!
0?????????@
 
)__inference_conv2d_4_layer_call_fn_401774a458’5
.’+
)&
inputs?????????@
ͺ "!?????????@·
D__inference_conv2d_5_layer_call_and_return_conditional_losses_401913oCD8’5
.’+
)&
inputs?????????@
ͺ "/’,
%"
0?????????
 
)__inference_conv2d_5_layer_call_fn_401922bCD8’5
.’+
)&
inputs?????????@
ͺ ""?????????«
C__inference_dense_5_layer_call_and_return_conditional_losses_402292dmn3’0
)’&
$!
inputs??????????
ͺ ")’&

0??????????@
 
(__inference_dense_5_layer_call_fn_402301Wmn3’0
)’&
$!
inputs??????????
ͺ "??????????@­
C__inference_dense_6_layer_call_and_return_conditional_losses_402685f3’0
)’&
$!
inputs?????????
ͺ ")’&

0?????????@
 
(__inference_dense_6_layer_call_fn_402694Y3’0
)’&
$!
inputs?????????
ͺ "?????????@§
C__inference_dense_7_layer_call_and_return_conditional_losses_402880`0’-
&’#
!
inputs?????????
ͺ "&’#

0?????????
 
(__inference_dense_7_layer_call_fn_402889S0’-
&’#
!
inputs?????????
ͺ "?????????¦
C__inference_dense_8_layer_call_and_return_conditional_losses_402982_­?0’-
&’#
!
inputs?????????
ͺ "%’"

0?????????@
 ~
(__inference_dense_8_layer_call_fn_402991R­?0’-
&’#
!
inputs?????????
ͺ "?????????@₯
C__inference_dense_9_layer_call_and_return_conditional_losses_403084^Ό½/’,
%’"
 
inputs?????????@
ͺ "%’"

0?????????
 }
(__inference_dense_9_layer_call_fn_403093QΌ½/’,
%’"
 
inputs?????????@
ͺ "?????????¦
E__inference_flatten_1_layer_call_and_return_conditional_losses_402864]3’0
)’&
$!
inputs?????????@
ͺ "&’#

0?????????
 ~
*__inference_flatten_1_layer_call_fn_402869P3’0
)’&
$!
inputs?????????@
ͺ "?????????
H__inference_functional_3_layer_call_and_return_conditional_losses_399922ΈP%&,-./45;<=>CDJKLM^_ghefmnvwtu|}§¨₯¦­?Ά·΄΅Ό½=’:
3’0
&#
input_2?????????
p

 
ͺ "%’"

0?????????
 
H__inference_functional_3_layer_call_and_return_conditional_losses_400063ΈP%&,-./45;<=>CDJKLM^_hegfmnwtvu|}¨₯§¦­?·΄Ά΅Ό½=’:
3’0
&#
input_2?????????
p 

 
ͺ "%’"

0?????????
 
H__inference_functional_3_layer_call_and_return_conditional_losses_401085·P%&,-./45;<=>CDJKLM^_ghefmnvwtu|}§¨₯¦­?Ά·΄΅Ό½<’9
2’/
%"
inputs?????????
p

 
ͺ "%’"

0?????????
 
H__inference_functional_3_layer_call_and_return_conditional_losses_401361·P%&,-./45;<=>CDJKLM^_hegfmnwtvu|}¨₯§¦­?·΄Ά΅Ό½<’9
2’/
%"
inputs?????????
p 

 
ͺ "%’"

0?????????
 έ
-__inference_functional_3_layer_call_fn_400322«P%&,-./45;<=>CDJKLM^_ghefmnvwtu|}§¨₯¦­?Ά·΄΅Ό½=’:
3’0
&#
input_2?????????
p

 
ͺ "?????????έ
-__inference_functional_3_layer_call_fn_400580«P%&,-./45;<=>CDJKLM^_hegfmnwtvu|}¨₯§¦­?·΄Ά΅Ό½=’:
3’0
&#
input_2?????????
p 

 
ͺ "?????????ά
-__inference_functional_3_layer_call_fn_401478ͺP%&,-./45;<=>CDJKLM^_ghefmnvwtu|}§¨₯¦­?Ά·΄΅Ό½<’9
2’/
%"
inputs?????????
p

 
ͺ "?????????ά
-__inference_functional_3_layer_call_fn_401595ͺP%&,-./45;<=>CDJKLM^_hegfmnwtvu|}¨₯§¦­?·΄Ά΅Ό½<’9
2’/
%"
inputs?????????
p 

 
ͺ "?????????Τ
K__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_398027E’B
;’8
63
inputs'???????????????????????????
ͺ ";’8
1.
0'???????????????????????????
 «
0__inference_max_pooling1d_1_layer_call_fn_398033wE’B
;’8
63
inputs'???????????????????????????
ͺ ".+'???????????????????????????
$__inference_signature_wrapper_400707ΫP%&,-./45;<=>CDJKLM^_hegfmnwtvu|}¨₯§¦­?·΄Ά΅Ό½@’=
’ 
6ͺ3
1
input_2&#
input_2?????????"EͺB
@
tf_op_layer_Mul_1+(
tf_op_layer_Mul_1?????????Ύ
T__inference_tf_op_layer_ExpandDims_1_layer_call_and_return_conditional_losses_401601f4’1
*’'
%"
inputs?????????
ͺ ".’+
$!
0?????????
 
9__inference_tf_op_layer_ExpandDims_1_layer_call_fn_401606Y4’1
*’'
%"
inputs?????????
ͺ "!?????????©
M__inference_tf_op_layer_Mul_1_layer_call_and_return_conditional_losses_403099X/’,
%’"
 
inputs?????????
ͺ "%’"

0?????????
 
2__inference_tf_op_layer_Mul_1_layer_call_fn_403104K/’,
%’"
 
inputs?????????
ͺ "?????????½
Q__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_402056h9’6
/’,
*'
inputs?????????
ͺ "+’(
!
0?????????
 
6__inference_tf_op_layer_Reshape_1_layer_call_fn_402061[9’6
/’,
*'
inputs?????????
ͺ "?????????Ή
S__inference_tf_op_layer_Transpose_1_layer_call_and_return_conditional_losses_402067b4’1
*’'
%"
inputs?????????
ͺ "*’'
 
0?????????
 
8__inference_tf_op_layer_Transpose_1_layer_call_fn_402072U4’1
*’'
%"
inputs?????????
ͺ "?????????