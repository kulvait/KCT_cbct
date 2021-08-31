#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 16:26:42 2021

Code to generate convolution kernels code when I have separable d0 and d1 vectors such that d0 * d1 is 2D kernel and d0 * d0 * d1 is 3D kernel

@author: Vojtech Kulvait
"""
import numpy as np

def generateConvolution2D(d0, d1):
	dx=len(d0)
	dy=len(d1)
	sumstr=""
	for i in range(dx):
		if not sumstr.endswith("\n") and not len(sumstr)==0:			
			sumstr="%s\n"%(sumstr)
		for j in range(dy):
			if np.float32(d0[i]*d1[j]) != 0:
				term = "%sf*cube[%d][%d]"%(np.format_float_positional(np.float32(d0[i]*d1[j])), i, j)
				if np.float32(d0[i]*d1[j]) > 0:
					sumstr="%s +%s"%(sumstr, term)
				else:
					sumstr="%s %s"%(sumstr, term)
	return sumstr;


def generateBracket(plusKeys, minusKeys, key):
	outplus=""
	outminus=""
	if key in plusKeys:
		outplus="+".join(plusKeys[key])
	if key in minusKeys:
		outminus="-".join(minusKeys[key])
		outminus="-%s"%(outminus)
	if not outplus:
		return outminus
	elif not outminus:
		return outplus
	else:
		return "%s%s"%(outplus, outminus)

def generateConvolutionCompact2D(d0, d1):
	dx=len(d0)
	dy=len(d1)
	plusTerms={}
	minusTerms={}
	for i in range(dx):
		for j in range(dy):
			cubestr="cube[%d][%d]"%(i,j)
			convitem=np.float32(d0[i]*d1[j])
			if convitem != 0:
				if convitem > 0:
					if convitem not in plusTerms:
						plusTerms[convitem]=[]
					plusTerms[convitem].append(cubestr)
				else:
					if -convitem not in minusTerms:
						minusTerms[-convitem]=[]
					minusTerms[-convitem].append(cubestr)
	allkeys=[]
	allkeys.extend(plusTerms.keys())
	allkeys.extend(minusTerms.keys())
	allkeys=list(set(allkeys))#Just unique
	allstr=[]
	for key in allkeys:
		itm="%sf*(%s)"%(np.float32(key), generateBracket(plusTerms, minusTerms, key))			
		allstr.append(itm)
	return "+\n".join(allstr)

def generateConvolutionCompact3D(d0, d1, d2):
	dx=len(d0)
	dy=len(d1)
	dz=len(d2)
	plusTerms={}
	minusTerms={}
	for i in range(dx):
		for j in range(dy):
			for k in range(dz):
				cubestr="cube[%d][%d][%d]"%(i,j,k)
				convitem=np.float32(d0[i]*d1[j]*d2[k])
				if convitem != 0:
					if convitem > 0:
						if convitem not in plusTerms:
							plusTerms[convitem]=[]
						plusTerms[convitem].append(cubestr)
					else:
						if -convitem not in minusTerms:
							minusTerms[-convitem]=[]
						minusTerms[-convitem].append(cubestr)
	allkeys=[]
	allkeys.extend(plusTerms.keys())
	allkeys.extend(minusTerms.keys())
	allkeys=list(set(allkeys))#Just unique
	allstr=[]
	for key in allkeys:
		itm="%sf*(%s)"%(np.float32(key), generateBracket(plusTerms, minusTerms, key))			
		allstr.append(itm)
	return "+\n".join(allstr)

def generateConvolution3D(d0, d1, d2):
	dx=len(d0)
	dy=len(d1)
	dz=len(d2)
	sumstr=""
	for i in range(dx):
		for j in range(dy):
			if not sumstr.endswith("\n") and not len(sumstr)==0:			
				sumstr="%s\n"%(sumstr)
			for k in range(dz):
				if np.float32(d0[i]*d1[j]*d2[k]) != 0:
					term = "%sf*cube[%d][%d][%d]"%(np.format_float_positional(np.float32(d0[i]*d1[j]*d2[k])), i, j, k)
					if np.float32(d0[i]*d1[j]*d2[k]) > 0:
						sumstr="%s +%s"%(sumstr, term)
					else:
						sumstr="%s %s"%(sumstr, term)
	return sumstr;


def normalizeVectorL1(v):
	return v/np.linalg.norm(v, ord=1)

x=generateConvolution2D([1,2,1], [-1,0,1])#Sobel2D
print(x)
sobeld0=[1,2,1]
sobeld1=[-1,0,1]
sobelnd0=sobeld0/np.linalg.norm(sobeld0, ord=1)
sobelnd1=sobeld1/np.linalg.norm(sobeld1, ord=1)
gx=generateConvolutionCompact2D([-1,0,1], [1,2,1])#Sobel2Dy
gy=generateConvolutionCompact2D([1,2,1], [-1,0,1])#Sobel2Dy
#Scaled sobel so that the derivatives can be just scaled by voxel sizes
gx=generateConvolutionCompact2D(sobelnd1, sobelnd0)#Sobel2Dy
gy=generateConvolutionCompact2D(sobelnd0, sobelnd1)#Sobel2Dy
print("grad.x=%s;"%gx)
print("grad.y=%s;"%gy)
x=generateConvolution3D([1,2,1], [1,2,1], [-1,0,1])#Sobel3D
sx=generateConvolutionCompact3D(sobelnd1, sobelnd0, sobelnd0)#Sobel3Dy
sy=generateConvolutionCompact3D(sobelnd0, sobelnd1, sobelnd0)#Sobel3Dy
sz=generateConvolutionCompact3D(sobelnd0, sobelnd0, sobelnd1)#Sobel3Dz
print("grad.x=%s;"%sx)
print("grad.y=%s;"%sy)
print("grad.z=%s;"%sz)

faridd0=[0.229879,0.540242,0.229879]
faridd1=[-0.425287,0,0.425287]
nd0=normalizeVectorL1(faridd0)
nd1=normalizeVectorL1(faridd1)
gx=generateConvolutionCompact2D(nd1, nd0)
gy=generateConvolutionCompact2D(nd0, nd1)
print()
print("grad.x=%s;"%gx)
print("grad.y=%s;"%gy)
gx=generateConvolutionCompact3D(nd1, nd0, nd0)
gy=generateConvolutionCompact3D(nd0, nd1, nd0)
gz=generateConvolutionCompact3D(nd0, nd0, nd1)
print()
print("grad.x=%s;"%gx)
print()
print("grad.y=%s;"%gy)
print()
print("grad.z=%s;"%gz)

faridd0=[0.037659, 0.249153, 0.426375, 0.249153, 0.037659]
faridd1=[-0.109604,-0.276691, 0, 0.276691, 0.109604]
nd0=normalizeVectorL1(faridd0)
nd1=normalizeVectorL1(faridd1)
gx=generateConvolutionCompact2D(nd1, nd0)
gy=generateConvolutionCompact2D(nd0, nd1)
print()
print("grad.x=%s;"%gx)
print("grad.y=%s;"%gy)
gx=generateConvolutionCompact3D(nd1, nd0, nd0)
gy=generateConvolutionCompact3D(nd0, nd1, nd0)
gz=generateConvolutionCompact3D(nd0, nd0, nd1)
print()
print("grad.x=%s;"%gx)
print()
print("grad.y=%s;"%gy)
print()
print("grad.z=%s;"%gz)