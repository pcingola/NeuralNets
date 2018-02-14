package com.pcingola.neunet.ising;

import java.awt.Container;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JTextField;
import javax.swing.SwingConstants;

import com.pcingola.neunet.Network;

/**
 * A Simple Ising neural network demo
 * 
 * @author pcingola@sinectis.com
 */
public class IsingNeuNetDemo implements ActionListener {

	private static String BUTTON_CREATE_NETWORK = "Create network";
	private static String BUTTON_RUN = "Run";
	private static String BUTTON_STOP = "Stop";

	private static JFrame frame;
	private IsingDrawingArea drawingArea;
	private JLabel message;
	private Network network;
	private JButton runButton, createNetworkButton, stopButton;
	private boolean running = false;
	private JTextField tempMax, tempMin, numIterations, numNeuronsX, numNeuronsY;

	//-------------------------------------------------------------------------
	// Static methods
	//-------------------------------------------------------------------------

	/**
	 * Create the GUI and show it. For thread safety, this method should be
	 * invoked from the event-dispatching thread.
	 */
	private static void createAndShowGUI() {
		//Make sure we have nice window decorations.
		JFrame.setDefaultLookAndFeelDecorated(true);

		//Create and set up the window.
		frame = new JFrame("Ising Neural network Demo");
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

		//Set up the content pane.
		IsingNeuNetDemo controller = new IsingNeuNetDemo();
		controller.buildUserInterface(frame.getContentPane());

		// Create a neural network
		controller.buildNetwork();

		//Display the window.
		frame.pack();
		frame.setVisible(true);
	}

	/**
	 * Main 
	 * @param args
	 */
	public static void main(String[] args) {
		// Schedule a job for the event-dispatching thread: creating and showing this application's GUI.
		javax.swing.SwingUtilities.invokeLater(new Runnable() {

			public void run() {
				createAndShowGUI();
			}
		});
	}

	//-------------------------------------------------------------------------
	// Methods
	//-------------------------------------------------------------------------

	/**
	 * Action dispatcher
	 */
	public void actionPerformed(ActionEvent e) {
		if( e.getActionCommand().equals(BUTTON_RUN) ) {
			if( isRunning() ) {
				JOptionPane.showMessageDialog(frame, "Network already running!");
			} else {
				setRunning(true);
				message.setText("Network running");
				drawingArea.runNetwork();
			}
		} else if( e.getActionCommand().equals(BUTTON_STOP) ) {
			setRunning(false);
			message.setText("Network stopped");
		} else if( e.getActionCommand().equals(BUTTON_CREATE_NETWORK) ) {
			if( isRunning() ) {
				JOptionPane.showMessageDialog(frame, "Network already running. Stop it first");
			} else {
				network = null;
				buildNetwork();
				clear();
				showNetwork(false);
			}
		}
	}

	/**
	 * Build a neural network
	 */
	public void buildNetwork() {
		if( network == null ) {
			// Create a network
			IsingNetwork net = new IsingNetwork(getNumNeuronsX(), getNumNeuronsY());
			network = net;
		}
	}

	/**
	 * Builds User Interface
	 * 
	 * @param container
	 */
	private void buildUserInterface(Container container) {
		int rowNum = 0, colNum = 0;

		container.setLayout(new GridBagLayout());
		GridBagConstraints gbc = new GridBagConstraints();

		// Add an area to draw
		drawingArea = new IsingDrawingArea(this);
		gbc.fill = GridBagConstraints.BOTH;
		gbc.weightx = 1;
		gbc.weighty = 1;
		gbc.gridx = 0;
		gbc.gridy = rowNum++;
		gbc.gridwidth = 10;
		gbc.gridheight = 10;
		container.add(drawingArea, gbc);

		
		// New row
		rowNum+=10;
		colNum = 0;

		// Add a 'message' label
		message = new JLabel(); //, drawingArea, JLabel.LEFT );
		message.setText("Ok");
		gbc.fill = GridBagConstraints.CENTER;
		gbc.gridx = colNum++;
		gbc.gridy = rowNum;
		gbc.gridwidth = 10;
		gbc.gridheight = 1;
		gbc.weightx = 0;
		gbc.weighty = 0;
		container.add(message, gbc);

		// All other elements are 1x1 cells in the grid and have no weight (for resize)
		gbc.fill = GridBagConstraints.NONE;
		gbc.weightx = 0;
		gbc.weighty = 0;
		gbc.gridwidth = 1;
		gbc.gridheight = 1;

		// New row
		rowNum++;
		colNum = 0;

		// Number neurons (layer's X size)
		numNeuronsX = new JTextField(5);
		numNeuronsX.setText("100");
		JLabel numNeuronsXLabel = new JLabel("Size X:", SwingConstants.RIGHT);
		gbc.gridx = colNum++;
		gbc.gridy = rowNum;
		container.add(numNeuronsXLabel, gbc);
		gbc.gridx = colNum++;
		gbc.gridy = rowNum;
		container.add(numNeuronsX, gbc);
		numNeuronsXLabel.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));

		// Number neurons (layer's Y size)
		numNeuronsY = new JTextField(5);
		numNeuronsY.setText("100");
		JLabel numNeuronsYLabel = new JLabel("Size Y:", SwingConstants.RIGHT);
		gbc.gridx = colNum++;
		gbc.gridy = rowNum;
		container.add(numNeuronsYLabel, gbc);
		gbc.gridx = colNum++;
		gbc.gridy = rowNum;
		container.add(numNeuronsY, gbc);
		numNeuronsYLabel.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));

		// New row
		rowNum++;
		colNum = 0;

		// Temp max
		tempMax = new JTextField(5);
		tempMax.setText("1.0");
		JLabel tempMaxLabel = new JLabel("Temp max:", SwingConstants.RIGHT);
		gbc.gridx = colNum++;
		gbc.gridy = rowNum;
		container.add(tempMaxLabel, gbc);
		gbc.gridx = colNum++;
		gbc.gridy = rowNum;
		container.add(tempMax, gbc);
		tempMaxLabel.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));

		// Temp min
		tempMin = new JTextField(5);
		tempMin.setText("0.01");
		JLabel tempMinLabel = new JLabel("Temp min:", SwingConstants.RIGHT);
		gbc.gridx = colNum++;
		gbc.gridy = rowNum;
		container.add(tempMinLabel, gbc);
		gbc.gridx = colNum++;
		gbc.gridy = rowNum;
		container.add(tempMin, gbc);
		tempMinLabel.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));

		// New row
		rowNum++;
		colNum = 0;

		// Number of iterations
		numIterations = new JTextField(5);
		numIterations.setText("5000");
		JLabel numIterationsLabel = new JLabel("Iterations:", SwingConstants.RIGHT);
		gbc.gridx = colNum++;
		gbc.gridy = rowNum;
		container.add(numIterationsLabel, gbc);
		gbc.gridx = colNum++;
		gbc.gridy = rowNum++;
		container.add(numIterations, gbc);
		numIterationsLabel.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));

		// New row
		rowNum++;
		colNum = 0;
		
		// Add a button 'Create network'
		createNetworkButton = new JButton(BUTTON_CREATE_NETWORK);
		createNetworkButton.setMnemonic(KeyEvent.VK_C);
		gbc.gridx = colNum++;
		gbc.gridy = rowNum;
		container.add(createNetworkButton, gbc);
		createNetworkButton.addActionListener(this);

		// Add a button "Run"
		runButton = new JButton(BUTTON_RUN);
		runButton.setMnemonic(KeyEvent.VK_R);
		gbc.gridx = colNum++;
		gbc.gridy = rowNum;
		container.add(runButton, gbc);
		runButton.addActionListener(this);

		// Add a button "Stop"
		stopButton = new JButton(BUTTON_STOP);
		stopButton.setMnemonic(KeyEvent.VK_S);
		gbc.gridx = colNum++;
		gbc.gridy = rowNum++;
		container.add(stopButton, gbc);
		stopButton.addActionListener(this);
	}

	public void clear() {
		drawingArea.clear();
	}

	public Network getNetwork() {
		return network;
	}

	public int getNumIterations() {
		try {
			return Integer.parseInt(numIterations.getText());
		} catch(Exception e) {
			return 0;
		}
	}

	public int getNumNeuronsX() {
		try {
			return Integer.parseInt(numNeuronsX.getText());
		} catch(Exception e) {
			return 0;
		}
	}

	public int getNumNeuronsY() {
		try {
			return Integer.parseInt(numNeuronsY.getText());
		} catch(Exception e) {
			return 0;
		}
	}

	public double getTempMax() {
		try {
			return Double.parseDouble(tempMax.getText());
		} catch(Exception e) {
			return 0;
		}
	}

	public double getTempMin() {
		try {
			return Double.parseDouble(tempMin.getText());
		} catch(Exception e) {
			return 0;
		}
	}

	public boolean isRunning() {
		return running;
	}

	public void setMessage(String text) {
		message.setText(text);
	}

	public void setRunning(boolean running) {
		this.running = running;
	}

	public void showNetwork(boolean showLablel) {
		drawingArea.showNetwork();
		if( showLablel ) setMessage("Size: " + network.getLayer(0).getSizeX() + " x " + network.getLayer(0).getSizeY());
	}

}
