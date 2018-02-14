package com.pcingola.neunet;

import com.pcingola.neunet.ising.IsingNetwork;


public class Zzz {

	public static void main(String[] args) {
		IsingNetwork net = new IsingNetwork(10,10);
		System.out.println(net.toString());
		net.calc();
		System.out.println(net.toString());
	}

}
