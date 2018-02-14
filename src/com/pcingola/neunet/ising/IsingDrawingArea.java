package com.pcingola.neunet.ising;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;

import javax.swing.JComponent;

/**
 * A drawing area
 * 
 * @author pcingola@sinectis.com
 */
public class IsingDrawingArea extends JComponent {

	protected IsingNeuNetDemo controller;
	protected Dimension preferredSize;

	//-------------------------------------------------------------------------
	// Constructor
	//-------------------------------------------------------------------------

	public IsingDrawingArea(IsingNeuNetDemo controller) {
		this.controller = controller;
		setBackground(Color.WHITE);
		setOpaque(true);
		preferredSize = new Dimension(800, 600);
		setMinimumSize(preferredSize);
	}

	//-------------------------------------------------------------------------
	// Methods
	//-------------------------------------------------------------------------

	/**
	 * Clear screen
	 */
	public void clear() {
		paintComponent(this.getGraphics());
	}

	public Dimension getPreferredSize() {
		return preferredSize;
	}

	protected void paintComponent(Graphics g) {
		// Paint background if we're opaque.
		if( isOpaque() ) {
			g.setColor(getBackground());
			g.fillRect(0, 0, getWidth(), getHeight());
		}
	}

	/**
	 * Run a network
	 */
	protected void runNetwork() {
		new IsingNetworkThread(controller, controller.getNetwork());
	}

	/**
	 * Show network's weights patterns
	 */
	protected void showNetwork() {
		if( (controller.getNetwork() != null) && (controller.getNetwork().getLayer(0) != null) ) {
			int i, j, x, y;
			IsingLayer lay = (IsingLayer) controller.getNetwork().getLayer(0);
			int outs[][] = lay.getOutputs();

			Graphics g = getGraphics();

			int sizeX = lay.getSizeX();
			int sizeY = lay.getSizeY();
			int boxSizeX = Math.max(getWidth() / sizeX, 1);
			int boxSizeY = Math.max(getHeight() / sizeY, 1);

			for( i = 0; i < sizeX; i++ ) {
				for( j = 0; j < sizeY; j++ ) {
					if( outs[i][j] > 0 ) g.setColor(Color.GREEN);
					else g.setColor(Color.RED);
					x = i * boxSizeX;
					y = j * boxSizeY;
					g.fillRect(x, y, boxSizeX, boxSizeY);
				}
			}
		}
	}

}
